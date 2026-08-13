from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import input_audit, load_inputs, region_hour_matrix
from .energy import baseline_energy_metrics, dispatch_without_storage, facility_load
from .forecast import run_forecast
from .plots import (
    plot_architecture,
    plot_demand,
    plot_forecast,
    plot_gantt,
    plot_gpu_utilization,
    plot_policy_comparison,
    plot_scenarios,
    plot_storage_region,
)
from .scheduler import GreedyScheduler
from .storage import optimize_storage


def _dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x))


def _relative_change(before: dict, after: dict) -> dict:
    return {
        k + "_change_pct": 100 * (after[k] / before[k] - 1)
        for k in ["operating_cost_cny", "carbon_tco2", "peak_net_grid_import_mw", "mean_abs_ramp_mw"]
        if abs(before[k]) > 1e-12
    }


def run_all(data_dir: str | Path, output_dir: str | Path, config_path: str | Path, quick: bool = False) -> dict:
    started = time.time()
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    full = output_dir / "full"
    for p in [figures, tables, full]:
        p.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open(encoding="utf-8") as f:
        cfg = json.load(f)

    inputs = load_inputs(data_dir)
    audit = input_audit(inputs)
    _dump_json(audit, tables / "input_audit.json")

    # Q1: descriptive statistics, forecasting, and a feasibility-first schedule.
    descriptive = inputs.tasks.groupby(["SourceRegion", "TaskType"]).agg(
        TaskCount=("TaskID", "size"),
        GPU_Demand=("GPU_Demand", "sum"),
        GPUh=("GPUh", "sum"),
        MeanDuration_min=("EstimatedDuration_min", "mean"),
        P95GPU=("GPU_Demand", lambda x: x.quantile(0.95)),
    ).reset_index()
    descriptive.to_csv(tables / "q1_workload_statistics.csv", index=False)
    forecast, forecast_metrics = run_forecast(inputs.tasks, cfg["seed"])
    forecast.to_csv(tables / "q1_forecast_2376_2399.csv", index=False)
    forecast_metrics.to_csv(tables / "q1_forecast_metrics.csv", index=False)

    scfg = cfg["scheduler"]
    q1 = GreedyScheduler(
        inputs, mode="baseline",
        training_max_wait=scfg["training_max_wait_h"],
        batch_max_wait=scfg["batch_max_wait_h"],
        candidate_top_k=scfg["candidate_top_k"],
    ).run()
    q1.schedule.to_csv(full / "q1_schedule_all.csv", index=False)
    q1_final = q1.schedule[q1.schedule["ArrivalHour"].between(2376, 2399)]
    q1_final.to_csv(tables / "q1_schedule_final24.csv", index=False)
    _dump_json(q1.violations, tables / "q1_constraint_validation.json")

    # Q2: multiobjective task scheduling without storage re-optimization.
    basic_energy, basic_metrics = dispatch_without_storage(inputs, facility_load(inputs, q1.ai_it_load))
    q2 = GreedyScheduler(
        inputs, mode="carbon", carbon_price=300.0,
        training_max_wait=scfg["training_max_wait_h"],
        batch_max_wait=scfg["batch_max_wait_h"],
        candidate_top_k=scfg["candidate_top_k"],
        wait_weight=scfg["wait_weight"], latency_weight=scfg["latency_weight"],
    ).run()
    q2.schedule.to_csv(full / "q2_schedule_all.csv", index=False)
    q2.schedule[q2.schedule["ArrivalHour"].between(2376, 2399)].to_csv(tables / "q2_schedule_final24.csv", index=False)
    _dump_json(q2.violations, tables / "q2_constraint_validation.json")
    q2_energy, q2_metrics = dispatch_without_storage(inputs, facility_load(inputs, q2.ai_it_load))
    q2_comparison = pd.DataFrame([
        {"Policy": "Basic schedule", **basic_metrics},
        {"Policy": "Cost-carbon trade-off", **q2_metrics},
    ])
    q2_comparison.to_csv(tables / "q2_policy_comparison.csv", index=False)

    # Q3: fixed baseline load, storage/electricity co-optimization.
    pue = inputs.gpu.set_index("Region").loc[inputs.regions, "PUE"].to_numpy(float)
    fixed_it = region_hour_matrix(inputs, "Baseline_AI_IT_Load_MW") + region_hour_matrix(inputs, "NonAI_IT_Load_MW")
    fixed_facility = fixed_it * pue[:, None]
    q3_base = baseline_energy_metrics(inputs)
    _, q3_no_storage = dispatch_without_storage(inputs, fixed_facility)
    q3 = optimize_storage(inputs, fixed_facility, carbon_price=300.0, peak_penalty=800.0, ramp_penalty=5.0)
    q3.dispatch.to_csv(full / "q3_storage_dispatch_all.csv", index=False)
    q3_summary = pd.DataFrame([
        {"Policy": "Provided baseline", **q3_base},
        {"Policy": "Same-load no storage", **q3_no_storage},
        {"Policy": "Storage optimized", **q3.metrics, **_relative_change(q3_no_storage, q3.metrics)},
    ])
    q3_summary.to_csv(tables / "q3_storage_effects.csv", index=False)
    _dump_json(q3.solver, tables / "q3_solver_status.json")

    # Q4: fixed-mode decomposition with matched one-factor scenario axes.
    # Tariff mechanisms re-solve the task layer; carbon epsilon caps and
    # renewable stresses re-solve the exact energy/storage layer.
    price_mechanisms = cfg["q4"]["price_mechanisms"]
    carbon_caps = cfg["q4"]["carbon_cap_ratios"]
    renewable_scales = cfg["q4"]["renewable_scales"]
    if quick:
        scenario_keys = [("realtime", r, "level", 1.0) for r in renewable_scales]
    else:
        scenario_keys = []
        scenario_keys += [(p, 1.0, "level", 1.0) for p in price_mechanisms]
        # Carbon caps are evaluated under a stressed renewable level so the
        # unconstrained comparator has positive grid emissions.
        scenario_keys += [("realtime", 0.6, "level", c) for c in carbon_caps]
        scenario_keys += [("realtime", r, "level", 1.0) for r in renewable_scales]
        scenario_keys += [("realtime", 1.0, "volatile", 1.0)]
        scenario_keys = list(dict.fromkeys(scenario_keys))
    scenario_rows = []
    best_joint = None
    best_objective = np.inf
    schedule_cache = {"realtime": q2}
    all_energy_valid = True
    for price_mechanism, renewable_scale, renewable_pattern, carbon_cap_ratio in scenario_keys:
        if price_mechanism not in schedule_cache:
            schedule_cache[price_mechanism] = GreedyScheduler(
                inputs, mode="carbon", carbon_price=300.0, renewable_scale=1.0,
                training_max_wait=scfg["training_max_wait_h"],
                batch_max_wait=scfg["batch_max_wait_h"], candidate_top_k=scfg["candidate_top_k"],
                wait_weight=scfg["wait_weight"], latency_weight=scfg["latency_weight"],
                price_mechanism=price_mechanism,
            ).run()
        sched = schedule_cache[price_mechanism]
        load = facility_load(inputs, sched.ai_it_load)
        storage = optimize_storage(
            inputs, load, carbon_price=300.0, renewable_scale=float(renewable_scale),
            peak_penalty=800.0, ramp_penalty=5.0,
            price_mechanism=price_mechanism, renewable_pattern=renewable_pattern,
            carbon_cap_ratio=float(carbon_cap_ratio),
        )
        m = storage.metrics
        all_energy_valid = all_energy_valid and bool(m["valid"])
        row = {
            "PriceMechanism": price_mechanism,
            "CarbonCapRatio": carbon_cap_ratio,
            "RenewableScale": renewable_scale,
            "RenewablePattern": renewable_pattern,
            **m,
            "mean_wait_h": float(sched.schedule.loc[sched.schedule["Status"] == "scheduled", "Wait_h"].mean()),
            "p95_wait_h": float(sched.schedule.loc[sched.schedule["Status"] == "scheduled", "Wait_h"].quantile(.95)),
            "mean_latency_ms": float(sched.schedule.loc[sched.schedule["Status"] == "scheduled", "NetworkLatency_ms"].mean()),
            "schedule_valid": sched.violations["valid"],
        }
        scenario_rows.append(row)
        if renewable_scale == 1.0 and renewable_pattern == "level" and carbon_cap_ratio == 1.0:
            objective = m["operating_cost_cny"] + 300 * m["carbon_tco2"] + 800 * m["peak_net_grid_import_mw"]
            if objective < best_objective:
                best_objective = objective
                best_joint = (sched, storage, price_mechanism, renewable_scale, renewable_pattern, carbon_cap_ratio)
    scenarios = pd.DataFrame(scenario_rows).sort_values(["PriceMechanism", "RenewablePattern", "RenewableScale", "CarbonCapRatio"])
    scenarios.to_csv(tables / "q4_scenario_results.csv", index=False)
    if best_joint is not None:
        sched, storage, price_mechanism, renewable_scale, renewable_pattern, carbon_cap_ratio = best_joint
        sched.schedule[sched.schedule["ArrivalHour"].between(2376, 2399)].to_csv(tables / "q4_recommended_schedule_final24.csv", index=False)
        storage.dispatch[storage.dispatch["Hour"].between(2376, 2406)].to_csv(tables / "q4_recommended_energy_final24.csv", index=False)
        _dump_json({
            "PriceMechanism": price_mechanism, "CarbonCapRatio": carbon_cap_ratio,
            "RenewableScale": renewable_scale, "RenewablePattern": renewable_pattern,
            "metrics": storage.metrics,
        }, tables / "q4_recommendation.json")

    # Figure system required by the Mathodology density gate.
    plot_architecture(figures / "model_architecture.svg")
    plot_demand(inputs.tasks, figures / "q1_demand_profile.svg")
    plot_forecast(forecast, figures / "q1_forecast.svg")
    capacities = inputs.gpu.set_index("Region").loc[inputs.regions, "Available_GPU"].to_numpy(float)
    plot_gpu_utilization(q1.gpu_load, capacities, inputs.regions, figures / "q1_gpu_utilization.svg")
    plot_gantt(q1.schedule, figures / "q1_gantt_final24.svg")
    plot_policy_comparison(q2_comparison, figures / "q2_policy_comparison.svg")
    plot_policy_comparison(q3_summary.iloc[1:3], figures / "q3_storage_effects.svg")
    plot_storage_region(q3.dispatch, "RegionE", figures / "q3_regionE_storage.svg")
    plot_scenarios(scenarios, figures / "q4_scenarios.svg")

    manifest = {
        "runtime_seconds": time.time() - started,
        "input_audit": audit,
        "q1_valid": q1.violations["valid"],
        "q2_valid": q2.violations["valid"] and q2_metrics["valid"],
        "q3_valid": q3.metrics["valid"],
        "q4_all_schedules_valid": bool(scenarios["schedule_valid"].all()),
        "q4_all_energy_valid": bool(all_energy_valid),
        "quick_mode": quick,
    }
    _dump_json(manifest, output_dir / "run_manifest.json")
    return manifest
