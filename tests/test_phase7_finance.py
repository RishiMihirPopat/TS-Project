"""Phase 7 (P1) Acceptance Tests: Finance / Volatility Structural-Break Case Study (H3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsdomain.utils import PROJECT_ROOT


def test_phase7_pelt_break_recovery() -> None:
    """Verify PELT recovers at least 4 of the 6 labelled crisis dates within +/- 30 trading days."""
    breaks_path = PROJECT_ROOT / "results" / "analysis" / "finance_breaks.csv"
    assert breaks_path.exists(), "Missing results/analysis/finance_breaks.csv"

    df_b = pd.read_csv(breaks_path)
    assert len(df_b) == 6
    n_recovered = int(df_b["recovered_within_30d"].sum())
    assert n_recovered >= 4, f"Expected >= 4 crisis breaks recovered within +/- 30d, got {n_recovered}"


def test_phase7_garch_and_h3_contrast_outputs() -> None:
    """Verify GARCH(1,1)-t, EWMA(0.94), Chronos_small, and Moirai_small metrics and H3 block-bootstrap CIs."""
    analysis_dir = PROJECT_ROOT / "results" / "analysis"
    reg_path = analysis_dir / "finance_metrics_by_regime.csv"
    h3_path = analysis_dir / "finance_h3_contrast.csv"

    assert reg_path.exists(), "Missing finance_metrics_by_regime.csv"
    assert h3_path.exists(), "Missing finance_h3_contrast.csv"

    reg_df = pd.read_csv(reg_path)
    req_models = {"Naive", "EWMA_094", "GARCH_1_1_t", "Chronos_small", "Moirai_small"}
    assert req_models.issubset(set(reg_df["model"]))
    assert {1, 5, 20}.issubset(set(reg_df["horizon"]))
    assert {"stable", "post_break"}.issubset(set(reg_df["regime"]))
    assert np.isfinite(reg_df["mase"].to_numpy()).all()
    assert np.isfinite(reg_df["qlike"].to_numpy()).all()
    assert np.isfinite(reg_df["kupiec_lr_stat"].to_numpy()).all()

    h3_df = pd.read_csv(h3_path)
    assert len(h3_df) == 6  # 3 horizons x 2 regimes
    assert np.isfinite(h3_df["chronos_minus_garch_mean"].to_numpy()).all()
    assert np.isfinite(h3_df["moirai_minus_garch_mean"].to_numpy()).all()
    assert np.isfinite(h3_df["chronos_minus_moirai_ci95_low"].to_numpy()).all()
    assert np.isfinite(h3_df["chronos_minus_moirai_ci95_high"].to_numpy()).all()
