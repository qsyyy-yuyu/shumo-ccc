from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .data import Inputs, region_hour_matrix
from .energy import energy_metrics, price_matrices, renewable_matrix


VARS = ["grid_load", "grid_charge", "renewable_load", "renewable_charge", "discharge", "sell", "curtail", "soc", "ramp"]


@dataclass
class StorageResult:
    dispatch: pd.DataFrame
    metrics: dict
    solver: dict


def optimize_storage(
    inputs: Inputs,
    load: np.ndarray,
    carbon_price: float = 300.0,
    peak_penalty: float = 800.0,
    ramp_penalty: float = 5.0,
    renewable_scale: float = 1.0,
    degradation_cost: float = 2.0,
    price_mechanism: str = "realtime",
    renewable_pattern: str = "level",
    carbon_cap_ratio: float | None = None,
) -> StorageResult:
    price, sell_price = price_matrices(inputs, price_mechanism)
    renewable = renewable_matrix(inputs, renewable_scale, renewable_pattern)
    regional_caps = [None] * len(inputs.regions)
    if carbon_cap_ratio is not None and carbon_cap_ratio < 1 - 1e-12:
        # A system epsilon cap is allocated proportionally to each region's
        # uncapped emissions for the same scenario. This preserves separability
        # while making the summed cap exactly carbon_cap_ratio times the
        # matched unconstrained comparator.
        base_frames = []
        minimum_frames = []
        for j, _ in enumerate(inputs.regions):
            base_frame, _ = _optimize_region(
                inputs, j, load[j], carbon_price, peak_penalty, ramp_penalty,
                renewable[j], price[j], sell_price[j], degradation_cost, None,
            )
            # A very large carbon price is a deterministic proxy for the
            # minimum-emission endpoint under the same physical constraints.
            minimum_frame, _ = _optimize_region(
                inputs, j, load[j], 1e9, 0.0, 0.0,
                renewable[j], price[j], sell_price[j], degradation_cost, None,
            )
            base_frames.append(base_frame)
            minimum_frames.append(minimum_frame)
        base_total = sum(float(x["Carbon_tCO2"].sum()) for x in base_frames)
        min_total = sum(float(x["Carbon_tCO2"].sum()) for x in minimum_frames)
        # ``carbon_cap_ratio`` is an actual multiplier of the matched
        # unconstrained emissions baseline. If a requested ratio lies below
        # the physical minimum, the minimum is used and reported explicitly.
        system_target = max(min_total, float(carbon_cap_ratio) * base_total)
        reducible = [
            max(0.0, float(b["Carbon_tCO2"].sum() - m["Carbon_tCO2"].sum()))
            for b, m in zip(base_frames, minimum_frames)
        ]
        total_reducible = sum(reducible)
        for j, (base_frame, minimum_frame) in enumerate(zip(base_frames, minimum_frames)):
            b = float(base_frame["Carbon_tCO2"].sum())
            m = float(minimum_frame["Carbon_tCO2"].sum())
            required_reduction = max(0.0, base_total - system_target) * (reducible[j] / max(total_reducible, 1e-12))
            regional_caps[j] = max(m, b - required_reduction) + 1e-7
    all_frames = []
    solver_rows = []
    for j, region in enumerate(inputs.regions):
        frame, info = _optimize_region(
            inputs, j, load[j], carbon_price, peak_penalty, ramp_penalty,
            renewable[j], price[j], sell_price[j], degradation_cost, regional_caps[j],
        )
        all_frames.append(frame)
        solver_rows.append({"Region": region, **info})
    dispatch = pd.concat(all_frames, ignore_index=True)
    metrics = energy_metrics(dispatch, renewable)
    metrics["simultaneous_charge_discharge_hours"] = int(
        ((dispatch["ChargePower_MW"] > 1e-5) & (dispatch["DischargePower_MW"] > 1e-5)).sum()
    )
    metrics["terminal_soc_min_margin_mwh"] = float(
        min(
            dispatch.loc[dispatch["Hour"] == 2406].set_index("Region")["SOC_MWh"]
            - inputs.storage.set_index("Region")["InitialSOC_MWh"]
        )
    )
    metrics["valid"] = bool(
        all(x["success"] for x in solver_rows)
        and metrics["simultaneous_charge_discharge_hours"] == 0
        and metrics["terminal_soc_min_margin_mwh"] >= -1e-5
    )
    metrics["carbon_cap_ratio"] = 1.0 if carbon_cap_ratio is None else float(carbon_cap_ratio)
    if carbon_cap_ratio is not None and carbon_cap_ratio < 1 - 1e-12:
        metrics["carbon_cap_tco2"] = float(sum(x or 0.0 for x in regional_caps))
        metrics["carbon_cap_slack_tco2"] = metrics["carbon_cap_tco2"] - metrics["carbon_tco2"]
        metrics["valid"] = bool(metrics["valid"] and metrics["carbon_cap_slack_tco2"] >= -1e-5)
    return StorageResult(dispatch, metrics, {"regions": solver_rows})


def _optimize_region(inputs, j, load, carbon_price, peak_penalty, ramp_penalty, avail, price, sell_price, degradation_cost, carbon_cap_tco2):
    region = inputs.regions[j]
    T = inputs.hours
    n_block = len(VARS)
    peak_idx = n_block * T
    n = peak_idx + 1

    def idx(name, t):
        return VARS.index(name) * T + t

    rt = inputs.region_time[inputs.region_time["Region"] == region].sort_values("Hour")
    ci = rt["CarbonIntensity_tCO2_per_MWh"].to_numpy(float)
    st = inputs.storage.set_index("Region").loc[region]

    c = np.zeros(n)
    c[[idx("grid_load", t) for t in range(T)]] = price + carbon_price * ci
    c[[idx("grid_charge", t) for t in range(T)]] = price + carbon_price * ci + degradation_cost
    c[[idx("renewable_charge", t) for t in range(T)]] = degradation_cost
    c[[idx("discharge", t) for t in range(T)]] = degradation_cost
    c[[idx("sell", t) for t in range(T)]] = -sell_price
    c[[idx("ramp", t) for t in range(T)]] = ramp_penalty
    c[peak_idx] = peak_penalty

    eq_rows = 3 * T
    Aeq = lil_matrix((eq_rows, n), dtype=float)
    beq = np.zeros(eq_rows)
    eta_c = float(st["ChargeEfficiency"])
    eta_d = float(st["DischargeEfficiency"])
    initial = float(st["InitialSOC_MWh"])
    for t in range(T):
        # Facility load balance.
        Aeq[t, idx("grid_load", t)] = 1
        Aeq[t, idx("renewable_load", t)] = 1
        Aeq[t, idx("discharge", t)] = 1
        beq[t] = load[t]
        # Renewable allocation balance.
        row = T + t
        for name in ["renewable_load", "renewable_charge", "sell", "curtail"]:
            Aeq[row, idx(name, t)] = 1
        beq[row] = avail[t]
        # End-of-hour SOC convention.
        row = 2 * T + t
        Aeq[row, idx("soc", t)] = 1
        if t > 0:
            Aeq[row, idx("soc", t - 1)] = -1
            beq[row] = 0
        else:
            beq[row] = initial
        Aeq[row, idx("renewable_charge", t)] = -eta_c
        Aeq[row, idx("grid_charge", t)] = -eta_c
        Aeq[row, idx("discharge", t)] = 1 / eta_d

    # import, charge, peak, ramp(+/-), and terminal constraints.
    extra = int(carbon_cap_tco2 is not None)
    Aub = lil_matrix((5 * T + 1 + extra, n), dtype=float)
    bub = np.zeros(5 * T + 1 + extra)
    max_import = float(st["MaxGridImport_MW"])
    max_export = min(float(st["SellLimit_MW"]), float(st.get("MaxGridExport_MW", st["SellLimit_MW"])))
    max_charge = float(st["MaxChargePower_MW"])
    for t in range(T):
        row = t
        Aub[row, idx("grid_load", t)] = 1
        Aub[row, idx("grid_charge", t)] = 1
        bub[row] = max_import

        row = T + t
        Aub[row, idx("grid_charge", t)] = 1
        Aub[row, idx("renewable_charge", t)] = 1
        bub[row] = max_charge

        row = 2 * T + t
        Aub[row, idx("grid_load", t)] = 1
        Aub[row, idx("grid_charge", t)] = 1
        Aub[row, idx("sell", t)] = -1
        Aub[row, peak_idx] = -1
        bub[row] = 0

        # |net_t - net_{t-1}| <= ramp_t; t=0 compares against zero.
        row = 3 * T + t
        for name, coeff in [("grid_load", 1), ("grid_charge", 1), ("sell", -1)]:
            Aub[row, idx(name, t)] = coeff
            if t > 0:
                Aub[row, idx(name, t - 1)] = -coeff
        Aub[row, idx("ramp", t)] = -1

        row = 4 * T + t
        for name, coeff in [("grid_load", -1), ("grid_charge", -1), ("sell", 1)]:
            Aub[row, idx(name, t)] = coeff
            if t > 0:
                Aub[row, idx(name, t - 1)] = -coeff
        Aub[row, idx("ramp", t)] = -1

    Aub[5 * T, idx("soc", T - 1)] = -1
    bub[5 * T] = -initial
    if carbon_cap_tco2 is not None:
        row = 5 * T + 1
        for t in range(T):
            Aub[row, idx("grid_load", t)] = ci[t]
            Aub[row, idx("grid_charge", t)] = ci[t]
        bub[row] = carbon_cap_tco2

    bounds = []
    for name in VARS:
        for t in range(T):
            if name == "grid_load" or name == "grid_charge":
                bounds.append((0, max_import))
            elif name == "renewable_load":
                bounds.append((0, min(load[t], avail[t])))
            elif name == "renewable_charge":
                bounds.append((0, min(max_charge, avail[t])))
            elif name == "discharge":
                bounds.append((0, float(st["MaxDischargePower_MW"])))
            elif name == "sell":
                bounds.append((0, max_export))
            elif name == "curtail":
                bounds.append((0, avail[t]))
            elif name == "soc":
                bounds.append((float(st["MinSOC_MWh"]), float(st["StorageCapacity_MWh"])))
            else:
                bounds.append((0, None))
    bounds.append((0, None))

    result = linprog(
        c,
        A_ub=Aub.tocsr(), b_ub=bub,
        A_eq=Aeq.tocsr(), b_eq=beq,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"Storage LP failed for {region}: {result.message}")

    x = result.x
    out = pd.DataFrame({"Region": region, "Hour": np.arange(T)})
    for name in VARS:
        out[name] = x[VARS.index(name) * T : (VARS.index(name) + 1) * T]
    out["FacilityLoad_MW"] = load
    out["GridPurchase_MW"] = out["grid_load"] + out["grid_charge"]
    out["GridSell_MW"] = out["sell"]
    out["RenewableToLoad_MW"] = out["renewable_load"]
    out["RenewableCharge_MW"] = out["renewable_charge"]
    out["ChargePower_MW"] = out["renewable_charge"] + out["grid_charge"]
    out["DischargePower_MW"] = out["discharge"]
    out["Curtailment_MW"] = out["curtail"]
    out["SOC_MWh"] = out["soc"]
    out["NetGridImport_MW"] = out["GridPurchase_MW"] - out["GridSell_MW"]
    out["Cost_CNY"] = out["GridPurchase_MW"] * price - out["GridSell_MW"] * sell_price
    out["Carbon_tCO2"] = out["GridPurchase_MW"] * ci
    keep = [
        "Region", "Hour", "FacilityLoad_MW", "GridPurchase_MW", "GridSell_MW",
        "RenewableToLoad_MW", "RenewableCharge_MW", "ChargePower_MW",
        "DischargePower_MW", "Curtailment_MW", "SOC_MWh", "NetGridImport_MW",
        "Cost_CNY", "Carbon_tCO2",
    ]
    out = out[keep]
    return out, {
        "success": True,
        "status": int(result.status),
        "objective": float(result.fun),
        "iterations": int(result.nit),
    }
