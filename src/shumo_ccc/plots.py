from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .figqa import assert_no_overlap


COLORS = {
    "AITraining": "#0072B2",
    "BatchInference": "#E69F00",
    "RealTimeInference": "#009E73",
}


def plot_architecture(path: Path):
    fig, ax = plt.subplots(figsize=(8.2, 3.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    nodes = [
        (0.4, 2.45, 1.8, 0.8, "Tasks\narrival & SLA", "#DDEBF7"),
        (0.4, 0.75, 1.8, 0.8, "Energy\nprice & carbon", "#E2F0D9"),
        (3.2, 1.6, 2.3, 1.0, "Feasible candidate\nregion-start set", "#FFF2CC"),
        (6.3, 1.6, 1.8, 1.0, "Task scheduling\nmaster problem", "#FCE4D6"),
        (8.7, 1.6, 1.0, 1.0, "Storage\nLP", "#E4DFEC"),
    ]
    for x, y, w, h, label, color in nodes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#555555")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for a, b in [((2.2, 2.85), (3.2, 2.2)), ((2.2, 1.15), (3.2, 2.0)), ((5.5, 2.1), (6.3, 2.1)), ((8.1, 2.1), (8.7, 2.1))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=12, color="#555555"))
    ax.text(5.0, 3.45, "Compute-power-storage decomposition", ha="center", va="center", fontsize=13, fontweight="bold")
    _finish(fig, path)


def _finish(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_overlap(fig)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".svg":
        fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_demand(tasks: pd.DataFrame, path: Path):
    hourly = tasks.groupby(["ArrivalHour", "TaskType"])["GPUh"].sum().unstack(fill_value=0).reindex(range(2400), fill_value=0)
    smooth = hourly.rolling(24, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8.2, 3.7), layout="constrained")
    for col in smooth:
        ax.plot(smooth.index, smooth[col], label=col, color=COLORS[col], linewidth=1.3)
    ax.set(xlabel="Hour", ylabel="24-hour rolling mean (GPUh/h)", title="Workload demand is dominated by AI training")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def plot_forecast(test: pd.DataFrame, path: Path):
    agg = test.groupby("Hour")[["GPUh", "HGBR", "Seasonal24"]].sum()
    fig, ax = plt.subplots(figsize=(8.2, 3.7), layout="constrained")
    ax.plot(agg.index, agg["GPUh"], marker="o", label="Actual", color="#111111")
    ax.plot(agg.index, agg["HGBR"], marker="s", label="HGBR", color="#0072B2")
    ax.plot(agg.index, agg["Seasonal24"], linestyle="--", label="Seasonal-24", color="#D55E00")
    ax.set(xlabel="Hour", ylabel="Arriving GPUh", title="Final 24-hour workload forecast")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def plot_gpu_utilization(gpu_load: np.ndarray, capacities: np.ndarray, regions: list[str], path: Path):
    util = 100 * gpu_load[:, 2376:2406] / capacities[:, None]
    fig, ax = plt.subplots(figsize=(8.2, 3.8), layout="constrained")
    im = ax.imshow(util, aspect="auto", cmap="viridis", vmin=0, vmax=max(80, float(util.max())))
    ax.set_yticks(range(len(regions)), regions)
    ax.set_xticks(range(0, 30, 3), [str(2376 + x) for x in range(0, 30, 3)])
    ax.set(xlabel="Hour", title="Regional GPU utilization: final 24 hours plus closure")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("GPU utilization (%)")
    _finish(fig, path)


def plot_gantt(schedule: pd.DataFrame, path: Path):
    s = schedule[schedule["ArrivalHour"].between(2376, 2399) & schedule["Status"].eq("scheduled")].copy()
    # Showing every task as a thin segment preserves completeness while sorting
    # by region and start keeps the 538-row chart readable.
    s = s.sort_values(["AssignedRegion", "StartHour", "TaskType", "GPU_Demand"]).reset_index(drop=True)
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.2), sharex=True, layout="constrained")
    for ax, (region, part) in zip(axes.flat, s.groupby("AssignedRegion", sort=True)):
        part = part.reset_index(drop=True)
        for y, row in part.iterrows():
            ax.barh(y, row["Duration_h"], left=row["StartHour"], height=0.8, color=COLORS[row["TaskType"]], linewidth=0)
        ax.set_title(f"{region} ({len(part)} tasks)", fontsize=10)
        ax.set_ylabel("Task order")
        ax.grid(axis="x", alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Execution time (hour)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, list(COLORS), loc="lower center", ncol=3, frameon=False)
    _finish(fig, path)


def plot_policy_comparison(summary: pd.DataFrame, path: Path):
    metrics = ["Operating cost", "Carbon", "Peak import", "Mean ramp"]
    cols = ["operating_cost_cny", "carbon_tco2", "peak_net_grid_import_mw", "mean_abs_ramp_mw"]
    base = summary.iloc[0][cols].astype(float).to_numpy()
    denom = np.where(np.abs(base) > 1e-12, np.abs(base), 1.0)
    values = summary[cols].astype(float).to_numpy() / denom * 100
    x = np.arange(len(metrics))
    width = 0.8 / len(summary)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), layout="constrained")
    for i, row in summary.reset_index(drop=True).iterrows():
        ax.bar(x - 0.4 + width / 2 + i * width, values[i], width, label=row["Policy"])
    ax.axhline(100, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Index (baseline = 100)")
    ax.set_title("Cost-carbon-peak-ramp trade-offs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=min(3, len(summary)), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def plot_storage_region(dispatch: pd.DataFrame, region: str, path: Path):
    d = dispatch[(dispatch["Region"] == region) & dispatch["Hour"].between(2352, 2406)]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.0), sharex=True, layout="constrained")
    axes[0].plot(d["Hour"], d["NetGridImport_MW"], color="#0072B2", label="Net import")
    axes[0].plot(d["Hour"], d["FacilityLoad_MW"], color="#555555", linestyle="--", label="Facility load")
    axes[0].set_ylabel("Power (MW)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(d["Hour"], d["SOC_MWh"], color="#009E73")
    axes[1].fill_between(d["Hour"], 0, d["SOC_MWh"], color="#009E73", alpha=0.15)
    axes[1].set(xlabel="Hour", ylabel="SOC (MWh)", title=f"{region} storage trajectory")
    axes[1].grid(axis="y", alpha=0.25)
    _finish(fig, path)


def plot_scenarios(scenarios: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), layout="constrained")
    s = scenarios.copy().reset_index(drop=True)
    labels = [
        f"{r.PriceMechanism[:2]} / R{r.RenewableScale:.1f} / C{r.CarbonCapRatio:.1f}"
        + (" / vol" if r.RenewablePattern == "volatile" else "")
        for r in s.itertuples()
    ]
    colors = ["#D55E00" if x < 0.999 else "#0072B2" for x in s["CarbonCapRatio"]]
    axes[0].scatter(s["carbon_tco2"], s["operating_cost_cny"] / 1e6, c=colors, s=45)
    axes[0].set(xlabel="Carbon emissions (tCO2)", ylabel="Net operating cost (million CNY)", title="Cost-carbon scenarios")
    axes[0].grid(alpha=0.25)
    x = np.arange(len(s))
    axes[1].bar(x, s["renewable_utilization"] * 100, color="#009E73", alpha=.85)
    axes[1].set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
    axes[1].set(ylabel="Renewable utilization incl. exports (%)", title="Scenario resource use")
    axes[1].grid(axis="y", alpha=0.25)
    _finish(fig, path)
