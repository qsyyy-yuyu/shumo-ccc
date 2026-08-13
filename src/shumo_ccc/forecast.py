from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def hourly_panel(tasks: pd.DataFrame, end_hour: int = 2399) -> pd.DataFrame:
    regions = sorted(tasks["SourceRegion"].unique())
    types = sorted(tasks["TaskType"].unique())
    idx = pd.MultiIndex.from_product(
        [range(end_hour + 1), regions, types], names=["Hour", "SourceRegion", "TaskType"]
    )
    grouped = tasks.groupby(["ArrivalHour", "SourceRegion", "TaskType"]).agg(
        TaskCount=("TaskID", "size"),
        GPU_Demand=("GPU_Demand", "sum"),
        GPUh=("GPUh", "sum"),
    )
    grouped.index.names = idx.names
    return grouped.reindex(idx, fill_value=0).reset_index()


def _features(panel: pd.DataFrame, target: str, observed_through: int | None = None) -> pd.DataFrame:
    out = panel.copy()
    h = out["Hour"].to_numpy()
    out["sin24"] = np.sin(2 * np.pi * h / 24)
    out["cos24"] = np.cos(2 * np.pi * h / 24)
    out["sin168"] = np.sin(2 * np.pi * h / 168)
    out["cos168"] = np.cos(2 * np.pi * h / 168)
    out = pd.get_dummies(out, columns=["SourceRegion", "TaskType"], dtype=float)

    source = panel[["Hour", "SourceRegion", "TaskType", target]].copy()
    if observed_through is not None:
        # Multi-step forecast at a fixed origin: targets inside the forecast
        # horizon are unavailable and must never leak through lag/rolling
        # features. Missing lag features are handled by the tree model.
        source.loc[source["Hour"] > observed_through, target] = np.nan
    source = source.sort_values(["SourceRegion", "TaskType", "Hour"])
    for lag in (1, 24, 168):
        source[f"lag{lag}"] = source.groupby(["SourceRegion", "TaskType"])[target].shift(lag)
    for window in (6, 24, 168):
        source[f"roll{window}"] = source.groupby(["SourceRegion", "TaskType"])[target].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    lag_cols = [c for c in source if c.startswith("lag") or c.startswith("roll")]
    out[lag_cols] = source.sort_index()[lag_cols]
    return out.fillna(0)


def _smape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(y) + np.abs(p)
    ratio = np.zeros_like(denom, dtype=float)
    np.divide(2 * np.abs(y - p), denom, out=ratio, where=denom > 1e-9)
    return float(np.mean(ratio))


def run_forecast(tasks: pd.DataFrame, seed: int = 20260813) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = hourly_panel(tasks)
    target = "GPUh"
    design = _features(panel, target)
    validation_design = _features(panel, target, observed_through=2351)
    test_design = _features(panel, target, observed_through=2375)
    feature_cols = [
        c for c in design.columns
        if c not in {"Hour", "TaskCount", "GPU_Demand", "GPUh"}
    ]

    train = design["Hour"] <= 2351
    validation = design["Hour"].between(2352, 2375)
    refit = design["Hour"] <= 2375
    test = design["Hour"].between(2376, 2399)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=24,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(design.loc[train, feature_cols], design.loc[train, target])
    val_pred = np.maximum(0, model.predict(validation_design.loc[validation, feature_cols]))

    model.fit(design.loc[refit, feature_cols], design.loc[refit, target])
    test_pred = np.maximum(0, model.predict(test_design.loc[test, feature_cols]))

    test_rows = panel.loc[test, ["Hour", "SourceRegion", "TaskType", target]].copy()
    test_rows["HGBR"] = test_pred
    lookup = panel.set_index(["Hour", "SourceRegion", "TaskType"])[target]
    for lag, name in [(24, "Seasonal24"), (168, "Seasonal168")]:
        test_rows[name] = [lookup.get((h - lag, r, typ), 0.0) for h, r, typ in test_rows[["Hour", "SourceRegion", "TaskType"]].itertuples(index=False, name=None)]

    val_actual = design.loc[validation, target].to_numpy()
    metric_rows = []
    for split, actual, predictions in [
        ("validation", val_actual, {"HGBR": val_pred}),
        ("test", test_rows[target].to_numpy(), {k: test_rows[k].to_numpy() for k in ["HGBR", "Seasonal24", "Seasonal168"]}),
    ]:
        for name, pred in predictions.items():
            metric_rows.append({
                "Split": split,
                "Model": name,
                "MAE_GPUh": mean_absolute_error(actual, pred),
                "RMSE_GPUh": mean_squared_error(actual, pred) ** 0.5,
                "sMAPE": _smape(actual, pred),
            })
    return test_rows, pd.DataFrame(metric_rows)
