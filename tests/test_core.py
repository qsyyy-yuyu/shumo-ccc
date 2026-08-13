from __future__ import annotations

import numpy as np
import pandas as pd

from shumo_ccc.forecast import _features
from shumo_ccc.scheduler import duration_overlap


def test_fractional_overlap_conserves_duration():
    hours, overlap = duration_overlap(10, 3.35)
    assert hours.tolist() == [10, 11, 12, 13]
    assert np.isclose(overlap.sum(), 3.35)
    assert np.all((overlap > 0) & (overlap <= 1))


def test_task_cannot_use_hour_2406_by_latest_start():
    duration = 6.65
    latest_start = int(np.floor(2406 - duration + 1e-10))
    hours, _ = duration_overlap(latest_start, duration)
    assert hours.max() <= 2405


def test_fixed_origin_features_do_not_read_forecast_window_targets():
    panel = pd.DataFrame({
        "Hour": range(10),
        "SourceRegion": ["RegionA"] * 10,
        "TaskType": ["AITraining"] * 10,
        "TaskCount": [1] * 10,
        "GPU_Demand": [1] * 10,
        "GPUh": np.arange(10, dtype=float),
    })
    a = _features(panel, "GPUh", observed_through=5)
    changed = panel.copy()
    changed.loc[changed["Hour"] > 5, "GPUh"] = 1_000_000
    b = _features(changed, "GPUh", observed_through=5)
    feature_cols = [c for c in a if c.startswith("lag") or c.startswith("roll")]
    assert np.allclose(a.loc[a.Hour > 5, feature_cols], b.loc[b.Hour > 5, feature_cols])
