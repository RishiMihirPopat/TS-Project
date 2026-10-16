"""Phase 2 acceptance tests: GIFT-Eval leaderboard Seasonal_Naive cross-check and classical baseline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsdomain.utils import validate_forecast_dataframe


def test_leaderboard_seasonal_naive_crosscheck() -> None:
    """Acceptance check: local Seasonal_Naive reproduces GIFT-Eval leaderboard MASE on >= 2 configs."""
    cc_path = Path("results/metrics/leaderboard_crosscheck.parquet")
    assert cc_path.exists(), "leaderboard_crosscheck.parquet missing; run `make baselines`."
    df = pd.read_parquet(cc_path)
    assert len(df) >= 2
    for _, row in df.iterrows():
        assert row["local_seasonal_naive_mase"] == pytest.approx(
            row["leaderboard_seasonal_naive_mase"], abs=1e-4
        ), f"Mismatch on {row['config']}: {row['local_seasonal_naive_mase']} vs {row['leaderboard_seasonal_naive_mase']}"


def test_classical_baselines_and_best_classical() -> None:
    """Acceptance check: all 4 core classical models ran on unseen & subset configs and produced valid parquets."""
    cb_path = Path("results/metrics/classical_baselines.parquet")
    best_local_path = Path("results/metrics/best_classical_local.parquet")
    best_lb_path = Path("results/metrics/best_classical_leaderboard.parquet")

    assert cb_path.exists()
    assert best_local_path.exists()
    assert best_lb_path.exists()

    cb_df = pd.read_parquet(cb_path)
    best_local_df = pd.read_parquet(best_local_path)
    best_lb_df = pd.read_parquet(best_lb_path)

    assert len(best_lb_df) == 97
    assert "unseen_mandi/W/short" in set(best_local_df["config"])
    assert "unseen_recency/D/short" in set(best_local_df["config"])

    # Verify best_classical_mase equals the minimum among the 4 core classical models
    for _, r in best_local_df.iterrows():
        expected_min = min(
            r["seasonal_naive_mase"],
            r["auto_ets_mase"],
            r["auto_theta_mase"],
            r["auto_arima_mase"],
        )
        assert r["best_classical_mase"] == pytest.approx(expected_min, abs=1e-9)

    # Verify forecast parquet + .meta.json schema for an unseen dataset
    sample_pq = Path("results/forecasts/AutoETS/unseen_mandi__W__short.parquet")
    sample_meta = Path("results/forecasts/AutoETS/unseen_mandi__W__short.parquet.meta.json")
    assert sample_pq.exists()
    assert sample_meta.exists()
    validate_forecast_dataframe(pd.read_parquet(sample_pq))
