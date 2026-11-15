"""Phase 5 (P1) Acceptance Tests: Native Quantiles & Time-Ordered Split Conformal Calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsdomain.metrics.conformal import (
    apply_conformal_to_forecast_df,
    verify_synthetic_conformal_coverage,
)
from tsdomain.utils import PROJECT_ROOT, QUANTILE_COLS


def test_phase5_multiscale_cqr_normalization() -> None:
    """Verify scale-normalized CQR does not corrupt low-scale series when pooled with 1000x scale series."""
    rng = np.random.default_rng(42)
    rows = []
    for item_id, scale in [("temp_c", 10.0), ("solar_wh", 5000.0)]:
        for w_id in range(10):
            for h in range(1, 11):
                y = scale + rng.normal(0.0, 0.2 * scale)
                rows.append({
                    "item_id": item_id,
                    "window_id": w_id,
                    "horizon_step": h,
                    "y_true": y,
                    "mean": scale,
                    **{col: scale + (q - 0.5) * 0.1 * scale for col, q in zip(QUANTILE_COLS, np.linspace(0.1, 0.9, 9))},
                })
    df = pd.DataFrame(rows)
    _, post_df, q_hat = apply_conformal_to_forecast_df(df, method="cqr", alpha=0.20)
    assert 0.0 < q_hat < 1.0
    temp_post = post_df[post_df["item_id"] == "temp_c"]
    temp_width = float(np.mean(temp_post["q0.9"] - temp_post["q0.1"]))
    assert temp_width < 20.0, f"Low-scale series width corrupted by high-scale series: {temp_width}"


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
