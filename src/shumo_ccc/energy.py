from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Inputs, region_hour_matrix


def price_matrices(inputs: Inputs, mechanism: str = "realtime") -> tuple[np.ndarray, np.ndarray]:
    """Return buy/sell prices under reproducible tariff mechanisms.

    ``flat`` uses each region's all-hour mean, ``tou`` uses the supplied
    PricePeriod conditional mean, and ``realtime`` keeps the observed series.
    """
    buy = region_hour_matrix(inputs, "ElectricityPrice_CNY_per_MWh")
    sell = region_hour_matrix(inputs, "SellPrice_CNY_per_MWh")
    if mechanism == "realtime":
        return buy, sell
    if mechanism == "flat":
        return np.repeat(buy.mean(axis=1, keepdims=True), inputs.hours, axis=1), np.repeat(
            sell.mean(axis=1, keepdims=True), inputs.hours, axis=1
        )
    if mechanism == "tou":
        out_buy = np.zeros_like(buy)
        out_sell = np.zeros_like(sell)
        for j, region in enumerate(inputs.regions):
            rt = inputs.region_time[inputs.region_time["Region"] == region].sort_values("Hour")
            out_buy[j] = rt["PricePeriod"].map(rt.groupby("PricePeriod")["ElectricityPrice_CNY_per_MWh"].mean()).to_numpy(float)
            out_sell[j] = rt["PricePeriod"].map(rt.groupby("PricePeriod")["SellPrice_CNY_per_MWh"].mean()).to_numpy(float)
        return out_buy, out_sell
    raise ValueError(f"Unknown price mechanism: {mechanism}")


def renewable_matrix(inputs: Inputs, scale: float = 1.0, pattern: str = "level") -> np.ndarray:
    base = region_hour_matrix(inputs, "AvailableRenewable_MW") * scale
    if pattern == "level":
        return base
    if pattern == "volatile":
        # Mean-preserving, spatially correlated daily fluctuation stress test.
        factor = 1.0 + 0.20 * np.sin(2 * np.pi * np.arange(inputs.hours) / 24.0)
        return base * factor[None, :]
    raise ValueError(f"Unknown renewable pattern: {pattern}")


def facility_load(inputs: Inputs, ai_it_load: np.ndarray) -> np.ndarray:
    nonai = region_hour_matrix(inputs, "NonAI_IT_Load_MW")
    pue = inputs.gpu.set_index("Region").loc[inputs.regions, "PUE"].to_numpy(float)
    return (nonai + ai_it_load) * pue[:, None]


def dispatch_without_storage(
    inputs: Inputs,
    load: np.ndarray,
    renewable_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    available = renewable_matrix(inputs, renewable_scale)
    price, sell_price = price_matrices(inputs, "realtime")
    carbon_intensity = region_hour_matrix(inputs, "CarbonIntensity_tCO2_per_MWh")
    limits = inputs.storage.set_index("Region").loc[inputs.regions]
    sell_limit = np.minimum(limits["SellLimit_MW"], limits["MaxGridExport_MW"]).to_numpy(float)[:, None]
    import_limit = limits["MaxGridImport_MW"].to_numpy(float)[:, None]

    renewable_to_load = np.minimum(load, available)
    grid_purchase = np.maximum(load - renewable_to_load, 0)
    surplus = np.maximum(available - renewable_to_load, 0)
    grid_sell = np.minimum(surplus, sell_limit)
    curtailment = surplus - grid_sell
    import_over = np.maximum(grid_purchase - import_limit, 0)

    rows = []
    for j, region in enumerate(inputs.regions):
        for t in range(inputs.hours):
            rows.append({
                "Region": region,
                "Hour": t,
                "FacilityLoad_MW": load[j, t],
                "RenewableToLoad_MW": renewable_to_load[j, t],
                "GridPurchase_MW": grid_purchase[j, t],
                "GridSell_MW": grid_sell[j, t],
                "Curtailment_MW": curtailment[j, t],
                "NetGridImport_MW": grid_purchase[j, t] - grid_sell[j, t],
                "Cost_CNY": grid_purchase[j, t] * price[j, t] - grid_sell[j, t] * sell_price[j, t],
                "Carbon_tCO2": grid_purchase[j, t] * carbon_intensity[j, t],
            })
    frame = pd.DataFrame(rows)
    metrics = energy_metrics(frame, available)
    metrics["max_grid_import_violation_mw"] = float(import_over.max())
    metrics["valid"] = bool(import_over.max() <= 1e-6)
    return frame, metrics


def energy_metrics(frame: pd.DataFrame, renewable_available: np.ndarray | None = None) -> dict:
    net = frame.pivot(index="Hour", columns="Region", values="NetGridImport_MW")
    peak = net.max().max()
    ramp = net.diff().abs().stack().mean()
    if renewable_available is None:
        renewable_total = float((frame["RenewableToLoad_MW"] + frame.get("RenewableCharge_MW", 0) + frame["GridSell_MW"] + frame["Curtailment_MW"]).sum())
    else:
        renewable_total = float(renewable_available.sum())
    utilization = 1 - float(frame["Curtailment_MW"].sum()) / max(renewable_total, 1e-9)
    return {
        "operating_cost_cny": float(frame["Cost_CNY"].sum()),
        "carbon_tco2": float(frame["Carbon_tCO2"].sum()),
        "renewable_utilization": float(utilization),
        "peak_net_grid_import_mw": float(peak),
        "mean_abs_ramp_mw": float(ramp),
    }


def baseline_energy_metrics(inputs: Inputs) -> dict:
    rt = inputs.region_time.copy()
    frame = pd.DataFrame({
        "Region": rt["Region"],
        "Hour": rt["Hour"],
        "GridPurchase_MW": rt["GridPurchase_MW"],
        "GridSell_MW": rt["GridSell_MW"],
        "Curtailment_MW": rt["Curtailment_MW"],
        "NetGridImport_MW": rt["NetGridImport_MW"],
        "Cost_CNY": rt["GridPurchase_MW"] * rt["ElectricityPrice_CNY_per_MWh"] - rt["GridSell_MW"] * rt["SellPrice_CNY_per_MWh"],
        "Carbon_tCO2": rt["GridPurchase_MW"] * rt["CarbonIntensity_tCO2_per_MWh"],
    })
    renewable = region_hour_matrix(inputs, "AvailableRenewable_MW")
    return energy_metrics(frame, renewable)
