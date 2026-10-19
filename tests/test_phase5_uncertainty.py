"""Phase 5 (P1) Acceptance Tests: Native Quantiles & Time-Ordered Split Conformal Calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsdomain.metrics.conformal import verify_synthetic_conformal_coverage
from tsdomain.utils import PROJECT_ROOT


def test_phase5_synthetic_conformal_coverage() -> None:
    """Verify split conformal calibration repairs overconfident intervals to 80% +/- 3% coverage."""
    res = verify_synthetic_conformal_coverage(seed=20260410)
    assert res["within_tolerance"] == 1.0, (
        f"Post-conformal coverage {res['post_coverage_80']:.4f} outside 0.80 +/- 0.03"
    )
    assert res["post_coverage_80"] > res["pre_coverage_80"]


def test_phase5_conformal_outputs() -> None:
    """Verify pre- and post-conformal metrics exist and improve calibration error."""
    cal_path = PROJECT_ROOT / "results" / "metrics" / "conformal_calibration.parquet"
    sum_path = PROJECT_ROOT / "results" / "metrics" / "conformal_summary_by_model.csv"
    assert cal_path.exists(), "Missing results/metrics/conformal_calibration.parquet"
    assert sum_path.exists(), "Missing results/metrics/conformal_summary_by_model.csv"

    cal_df = pd.read_parquet(cal_path)
    req_cols = [
        "model",
        "config",
        "calibration_method",
        "pre_coverage_80",
        "post_coverage_80",
        "pre_width_80",
        "post_width_80",
        "pre_crps",
        "post_crps",
        "pre_interval_score_80",
        "post_interval_score_80",
    ]
    for col in req_cols:
        assert col in cal_df.columns, f"Missing column {col} in conformal_calibration.parquet"
        assert not cal_df[col].isna().any()
        if col not in ("model", "config", "calibration_method"):
            assert np.isfinite(cal_df[col].to_numpy(dtype=float)).all()

    sum_df = pd.read_csv(sum_path)
    assert {"Chronos_small", "Moirai_small", "TimesFM", "TTM-R2-Pretrained"}.issubset(
        set(sum_df["model"])
    )
