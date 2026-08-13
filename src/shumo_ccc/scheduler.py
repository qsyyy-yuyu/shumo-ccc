from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Inputs, region_hour_matrix
from .energy import price_matrices


@dataclass
class ScheduleResult:
    schedule: pd.DataFrame
    gpu_load: np.ndarray
    gpu_occupancy: np.ndarray
    ai_it_load: np.ndarray
    violations: dict


def duration_overlap(start: int, duration_h: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.ceil(duration_h - 1e-12))
    hours = np.arange(start, start + n, dtype=int)
    overlaps = np.clip(duration_h - np.arange(n), 0.0, 1.0)
    return hours, overlaps


class GreedyScheduler:
    """Deterministic feasible scheduler for large non-preemptive task sets.

    Real-time tasks are placed first at their arrival hour. Flexible tasks are
    then placed using a restricted candidate set and an explicit feasibility
    fallback. This implements a scalable decomposition heuristic for the exact
    time-indexed binary model documented in docs/model.md.
    """

    def __init__(
        self,
        inputs: Inputs,
        mode: str = "baseline",
        carbon_price: float = 300.0,
        renewable_scale: float = 1.0,
        training_max_wait: int = 168,
        batch_max_wait: int = 48,
        candidate_top_k: int = 6,
        wait_weight: float = 0.08,
        latency_weight: float = 0.08,
        price_mechanism: str = "realtime",
    ):
        self.inputs = inputs
        self.mode = mode
        self.regions = inputs.regions
        self.ridx = {r: i for i, r in enumerate(self.regions)}
        self.hours = inputs.hours
        self.gpu_capacity = inputs.gpu.set_index("Region").loc[self.regions, "Available_GPU"].to_numpy(float)
        self.max_it = inputs.gpu.set_index("Region").loc[self.regions, "Max_IT_Power_MW"].to_numpy(float)
        self.pue = inputs.gpu.set_index("Region").loc[self.regions, "PUE"].to_numpy(float)
        self.nonai = region_hour_matrix(inputs, "NonAI_IT_Load_MW")
        self.price, self.sell_price = price_matrices(inputs, price_mechanism)
        self.carbon = region_hour_matrix(inputs, "CarbonIntensity_tCO2_per_MWh")
        self.renewable = region_hour_matrix(inputs, "AvailableRenewable_MW") * renewable_scale
        sell = inputs.storage.set_index("Region").loc[self.regions, "SellLimit_MW"].to_numpy(float)
        self.sell_limit = sell[:, None]
        self.power = inputs.power.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].to_dict()
        self.latency = inputs.latency.set_index(["FromRegion", "ToRegion"])["NetworkLatency_ms"].to_dict()
        self.gpu_load = np.zeros((len(self.regions), self.hours), dtype=float)
        self.gpu_occupancy = np.zeros_like(self.gpu_load)
        self.ai_it_load = np.zeros_like(self.gpu_load)
        self.carbon_price = carbon_price
        self.training_max_wait = training_max_wait
        self.batch_max_wait = batch_max_wait
        self.candidate_top_k = candidate_top_k
        self.wait_weight = wait_weight
        self.latency_weight = latency_weight
        self.signal = self._build_signal()

    def _build_signal(self) -> dict[str, np.ndarray]:
        signals = {}
        base_facility = self.nonai * self.pue[:, None]
        surplus = np.maximum(self.renewable - base_facility, 0)
        free_curtailment = np.maximum(surplus - self.sell_limit, 0)
        for task_type, p in self.power.items():
            facility_per_gpu = self.pue[:, None] * p
            # Opportunity cost: curtailment is free, sellable renewable costs the
            # foregone feed-in revenue, and grid energy costs price + carbon.
            marginal = np.where(
                free_curtailment > facility_per_gpu,
                0.0,
                np.where(surplus > 0, self.sell_price, self.price + self.carbon_price * self.carbon),
            )
            scale = np.nanmedian(marginal[marginal > 0]) if np.any(marginal > 0) else 1.0
            signals[task_type] = marginal * facility_per_gpu / max(scale, 1e-9)
        return signals

    def eligible_regions(self, source: str, max_latency: float) -> list[int]:
        return [j for j, r in enumerate(self.regions) if self.latency[(source, r)] <= max_latency + 1e-9]

    def _feasible(self, j: int, task: pd.Series, start: int) -> tuple[bool, np.ndarray, np.ndarray]:
        hours, overlap = duration_overlap(start, float(task["Duration_h"]))
        if len(hours) == 0 or hours[-1] >= 2406 or hours[-1] >= self.hours:
            return False, hours, overlap
        gpu_add = float(task["GPU_Demand"]) * overlap
        it_add = gpu_add * self.power[task["TaskType"]]
        # Attachment 1 fixes the hourly compute-capacity convention as
        # equivalent GPU-hour, so fractional-hour tasks consume g_i * overlap
        # rather than a full g_i for every touched hour.
        ok = bool(
            np.all(self.gpu_load[j, hours] + gpu_add <= self.gpu_capacity[j] + 1e-8)
            and np.all(self.nonai[j, hours] + self.ai_it_load[j, hours] + it_add <= self.max_it[j] + 1e-8)
        )
        return ok, hours, overlap

    def _place(self, j: int, task: pd.Series, start: int, hours: np.ndarray, overlap: np.ndarray) -> dict:
        gpu_add = float(task["GPU_Demand"]) * overlap
        it_add = gpu_add * self.power[task["TaskType"]]
        self.gpu_load[j, hours] += gpu_add
        self.gpu_occupancy[j, hours] += float(task["GPU_Demand"])
        self.ai_it_load[j, hours] += it_add
        return {
            "TaskID": int(task["TaskID"]),
            "TaskType": task["TaskType"],
            "SourceRegion": task["SourceRegion"],
            "AssignedRegion": self.regions[j],
            "ArrivalHour": int(task["ArrivalHour"]),
            "StartHour": int(start),
            "FinishTime": float(start + task["Duration_h"]),
            "GPU_Demand": int(task["GPU_Demand"]),
            "Duration_h": float(task["Duration_h"]),
            "Wait_h": int(start - task["ArrivalHour"]),
            "NetworkLatency_ms": float(self.latency[(task["SourceRegion"], self.regions[j])]),
            "Status": "scheduled",
        }

    def _candidate_starts(self, task: pd.Series, eligible: list[int]) -> list[int]:
        earliest = int(task["EarliestStartHour"])
        max_start = int(np.floor(float(task["LatestFinishHour"]) - float(task["Duration_h"]) + 1e-10))
        max_start = min(max_start, 2405)
        if self.mode == "baseline":
            return list(range(earliest, max_start + 1))

        horizon = self.training_max_wait if task["TaskType"] == "AITraining" else self.batch_max_wait
        policy_end = min(max_start, earliest + horizon)
        starts = set(range(earliest, min(policy_end, earliest + 8) + 1))
        for j in eligible:
            values = self.signal[task["TaskType"]][j, earliest : policy_end + 1]
            if values.size:
                k = min(self.candidate_top_k, values.size)
                for q in np.argpartition(values, k - 1)[:k]:
                    starts.add(earliest + int(q))
        starts.add(policy_end)
        return sorted(starts)

    def _score(self, j: int, task: pd.Series, start: int, hours: np.ndarray, overlap: np.ndarray) -> float:
        latency = self.latency[(task["SourceRegion"], self.regions[j])] / max(float(task["MaxLatency_ms"]), 1.0)
        wait_cap = self.training_max_wait if task["TaskType"] == "AITraining" else self.batch_max_wait
        wait = (start - int(task["ArrivalHour"])) / max(wait_cap, 1)
        utilization = np.max((self.gpu_load[j, hours] + float(task["GPU_Demand"]) * overlap) / self.gpu_capacity[j])
        if self.mode == "baseline":
            return start * 1e3 + latency + 0.1 * utilization
        energy = float(np.dot(overlap, self.signal[task["TaskType"]][j, hours]))
        return energy + self.wait_weight * wait + self.latency_weight * latency + 0.03 * utilization

    def _schedule_realtime(self, task: pd.Series) -> dict:
        start = int(task["ArrivalHour"])
        candidates = []
        for j in self.eligible_regions(task["SourceRegion"], float(task["MaxLatency_ms"])):
            ok, hours, overlap = self._feasible(j, task, start)
            if ok:
                score = self.latency[(task["SourceRegion"], self.regions[j])] + 5 * np.max(
                    (self.gpu_load[j, hours] + float(task["GPU_Demand"]) * overlap) / self.gpu_capacity[j]
                )
                candidates.append((score, j, hours, overlap))
        if not candidates:
            return {"TaskID": int(task["TaskID"]), "Status": "infeasible_realtime"}
        _, j, hours, overlap = min(candidates, key=lambda x: x[0])
        return self._place(j, task, start, hours, overlap)

    def _schedule_flexible(self, task: pd.Series) -> dict:
        eligible = self.eligible_regions(task["SourceRegion"], float(task["MaxLatency_ms"]))
        candidates = []
        for start in self._candidate_starts(task, eligible):
            for j in eligible:
                ok, hours, overlap = self._feasible(j, task, start)
                if ok:
                    candidates.append((self._score(j, task, start, hours, overlap), j, start, hours, overlap))
            if self.mode == "baseline" and candidates:
                break
        if not candidates and self.mode != "baseline":
            # Feasibility fallback scans the complete official window, so the
            # policy horizon never converts a feasible official task into a miss.
            earliest = int(task["EarliestStartHour"])
            max_start = int(np.floor(float(task["LatestFinishHour"]) - float(task["Duration_h"]) + 1e-10))
            for start in range(earliest, min(max_start, 2405) + 1):
                for j in eligible:
                    ok, hours, overlap = self._feasible(j, task, start)
                    if ok:
                        candidates.append((self._score(j, task, start, hours, overlap), j, start, hours, overlap))
                if candidates:
                    break
        if not candidates:
            return {"TaskID": int(task["TaskID"]), "Status": "infeasible_flexible"}
        _, j, start, hours, overlap = min(candidates, key=lambda x: x[0])
        return self._place(j, task, start, hours, overlap)

    def run(self) -> ScheduleResult:
        tasks = self.inputs.tasks.copy()
        real = tasks[tasks["TaskType"] == "RealTimeInference"].sort_values(
            ["ArrivalHour", "GPU_Demand"], ascending=[True, False]
        )
        flex = tasks[tasks["TaskType"] != "RealTimeInference"].copy()
        flex["Work"] = flex["GPU_Demand"] * flex["Duration_h"]
        flex = flex.sort_values(["LatestFinishHour", "ArrivalHour", "Work"], ascending=[True, True, False])

        records = [self._schedule_realtime(row) for _, row in real.iterrows()]
        records.extend(self._schedule_flexible(row) for _, row in flex.iterrows())
        schedule = pd.DataFrame(records).sort_values("TaskID").reset_index(drop=True)
        violations = validate_schedule(self.inputs, schedule, self.gpu_load, self.gpu_occupancy, self.ai_it_load)
        return ScheduleResult(schedule, self.gpu_load, self.gpu_occupancy, self.ai_it_load, violations)


def validate_schedule(inputs: Inputs, schedule: pd.DataFrame, gpu_load: np.ndarray, gpu_occupancy: np.ndarray, ai_it: np.ndarray) -> dict:
    complete = schedule["Status"].eq("scheduled") if "Status" in schedule else pd.Series(False, index=schedule.index)
    good = schedule.loc[complete].copy()
    caps = inputs.gpu.set_index("Region").loc[inputs.regions]
    nonai = region_hour_matrix(inputs, "NonAI_IT_Load_MW")
    max_gpu_over = float(np.max(gpu_load - caps["Available_GPU"].to_numpy(float)[:, None]))
    max_instant_gpu_over = float(np.max(gpu_occupancy - caps["Available_GPU"].to_numpy(float)[:, None]))
    max_it_over = float(np.max(nonai + ai_it - caps["Max_IT_Power_MW"].to_numpy(float)[:, None]))
    latency_over = float((good["NetworkLatency_ms"] - good["TaskID"].map(inputs.tasks.set_index("TaskID")["MaxLatency_ms"])).max()) if len(good) else 0.0
    realtime_wait = int(good.loc[good["TaskType"] == "RealTimeInference", "Wait_h"].abs().max()) if len(good) else 0
    closure_violation = int((good["FinishTime"] > 2406 + 1e-9).sum()) if len(good) else 0
    return {
        "tasks_total": int(len(inputs.tasks)),
        "tasks_scheduled": int(complete.sum()),
        "tasks_infeasible": int((~complete).sum()),
        "max_gpu_capacity_violation": max(0.0, max_gpu_over),
        "supplementary_instantaneous_gpu_overage": max(0.0, max_instant_gpu_over),
        "max_it_power_violation_mw": max(0.0, max_it_over),
        "max_latency_violation_ms": max(0.0, latency_over),
        "max_realtime_wait_h": realtime_wait,
        "finish_after_2406_count": closure_violation,
        "valid": bool(
            complete.all() and max_gpu_over <= 1e-6 and max_it_over <= 1e-6
            and latency_over <= 1e-6 and realtime_wait == 0 and closure_violation == 0
        ),
    }
