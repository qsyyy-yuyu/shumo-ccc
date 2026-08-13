from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Inputs:
    tasks: pd.DataFrame
    gpu: pd.DataFrame
    latency: pd.DataFrame
    power: pd.DataFrame
    region_time: pd.DataFrame
    storage: pd.DataFrame

    @property
    def regions(self) -> list[str]:
        return self.gpu["Region"].tolist()

    @property
    def hours(self) -> int:
        return int(self.region_time["Hour"].max()) + 1


FILES = {
    "tasks": "workload_trace.xlsx",
    "gpu": "GPU_information.xlsx",
    "latency": "network_latency.xlsx",
    "power": "power_mapping.xlsx",
    "region_time": "region_time_data.xlsx",
    "storage": "storage_information.xlsx",
}


def load_inputs(data_dir: str | Path) -> Inputs:
    data_dir = Path(data_dir)
    missing = [name for name in FILES.values() if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files in {data_dir}: {missing}")

    tasks = pd.read_excel(data_dir / FILES["tasks"], sheet_name="Sheet1")
    gpu = pd.read_excel(data_dir / FILES["gpu"], sheet_name="GPU中心基础情况")
    latency = pd.read_excel(data_dir / FILES["latency"], sheet_name="network_latency")
    power = pd.read_excel(data_dir / FILES["power"], sheet_name="任务功率映射")
    region_time = pd.read_excel(data_dir / FILES["region_time"], sheet_name="region_time_data")
    storage = pd.read_excel(data_dir / FILES["storage"], sheet_name="storage_information")

    tasks = tasks.copy()
    tasks["Duration_h"] = tasks["EstimatedDuration_min"] / 60.0
    tasks["GPUh"] = tasks["GPU_Demand"] * tasks["Duration_h"]

    for df, cols in [
        (tasks, ["TaskID", "ArrivalHour", "GPU_Demand", "EstimatedDuration_min"]),
        (gpu, ["Available_GPU", "Max_IT_Power_MW", "PUE"]),
        (region_time, ["Hour", "ElectricityPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh"]),
    ]:
        if df[cols].isna().any().any():
            raise ValueError(f"Missing required numeric value in columns {cols}")

    expected_hours = set(range(2407))
    for region, part in region_time.groupby("Region"):
        if set(part["Hour"].astype(int)) != expected_hours:
            raise ValueError(f"Region {region} does not contain exactly Hours 0..2406")

    return Inputs(tasks, gpu, latency, power, region_time, storage)


def input_audit(inputs: Inputs) -> dict:
    rt = inputs.region_time
    pue = inputs.gpu.set_index("Region")["PUE"]
    checks = {
        "metric_definition_source": "Attachment1.docx",
        "gpu_capacity_convention": "fractional_overlap_GPUh_per_hour",
        "carbon_definition": "GridPurchase_MWh_x_CarbonIntensity",
        "renewable_utilization_definition": "(direct+renewable_charge+export)/available",
        "task_count": int(len(inputs.tasks)),
        "region_count": int(len(inputs.regions)),
        "hour_min": int(rt["Hour"].min()),
        "hour_max": int(rt["Hour"].max()),
        "facility_identity_max_abs_mw": float(
            (rt["Total_Load_MW"] - rt["IT_Load_MW"] * rt["Region"].map(pue)).abs().max()
        ),
        "carbon_identity_max_abs_tco2": float(
            (rt["CarbonEmission_tCO2"] - rt["GridPurchase_MW"] * rt["CarbonIntensity_tCO2_per_MWh"]).abs().max()
        ),
        "renewable_balance_max_abs_mw": float(
            (
                rt["AvailableRenewable_MW"]
                - rt[["UsedRenewable_MW", "RenewableCharge_MW", "GridSell_MW", "Curtailment_MW"]].sum(axis=1)
            ).abs().max()
        ),
        "renewable_identical_across_regions": bool(
            (rt.pivot(index="Hour", columns="Region", values="AvailableRenewable_MW").nunique(axis=1) == 1).all()
        ),
    }
    checks["valid"] = bool(
        checks["facility_identity_max_abs_mw"] < 1e-3
        and checks["carbon_identity_max_abs_tco2"] < 1e-3
        and checks["renewable_balance_max_abs_mw"] < 1e-3
    )
    return checks


def region_hour_matrix(inputs: Inputs, column: str) -> np.ndarray:
    return (
        inputs.region_time.pivot(index="Hour", columns="Region", values=column)
        .reindex(index=range(inputs.hours), columns=inputs.regions)
        .to_numpy(dtype=float)
        .T
    )
