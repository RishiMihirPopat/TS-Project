"""Phase 6 Acceptance Tests: Hierarchical Mixed-Effects Models & Leave-Dataset-Out CV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsdomain.analysis.mixed_models import recover_planted_synthetic_effect
from tsdomain.utils import PROJECT_ROOT


def test_phase6_synthetic_planted_effect_recovery() -> None:
    """Verify the mixed-effects fitting pipeline recovers a planted beta_sim_distance = -0.15 within 95% CI."""
    res = recover_planted_synthetic_effect(true_beta_sim_distance=-0.15, seed=20260410)
    assert res["covered"] == 1.0, (
        f"Planted beta=-0.15 not covered by 95% CI [{res['ci95_low']:.4f}, {res['ci95_high']:.4f}] "
        f"(est={res['estimated_beta']:.4f})"
    )
    assert abs(res["estimated_beta"] - (-0.15)) < 0.03


def test_phase6_mixed_model_outputs_and_lodo_cv() -> None:
    """Verify M0..M3 comparison, REML coefficients + bootstrap CIs, H2 contrasts, and LODO CV outputs."""
    analysis_dir = PROJECT_ROOT / "results" / "analysis"
    comp_path = analysis_dir / "mixed_model_comparison.csv"
    coef_path = analysis_dir / "mixed_model_coefficients.csv"
    h2_path = analysis_dir / "h2_strata_contrasts.csv"
    cv_path = analysis_dir / "lodo_cv_summary.csv"

    assert comp_path.exists(), "Missing mixed_model_comparison.csv"
    assert coef_path.exists(), "Missing mixed_model_coefficients.csv"
    assert h2_path.exists(), "Missing h2_strata_contrasts.csv"
    assert cv_path.exists(), "Missing lodo_cv_summary.csv"

    comp_df = pd.read_csv(comp_path)
    for m_id in ["M0", "M1", "M2a", "M2b", "M2", "M3"]:
        assert m_id in comp_df["model_id"].values, f"Missing {m_id} in mixed_model_comparison.csv"
    assert np.isfinite(comp_df["AIC_ML"].to_numpy()).all()
    assert np.isfinite(comp_df["BIC_ML"].to_numpy()).all()
    assert np.isfinite(comp_df["R2_marginal"].to_numpy()).all()
    assert np.isfinite(comp_df["R2_conditional"].to_numpy()).all()

    coef_df = pd.read_csv(coef_path)
    m2_coefs = coef_df[coef_df["model_id"] == "M2"]
    for term in ["sim_binary", "z_sim_share", "z_sim_distance"]:
        assert term in m2_coefs["term"].values, f"Missing {term} in M2 coefficients"
    assert np.isfinite(m2_coefs["boot_ci95_low"].to_numpy()).all()
    assert np.isfinite(m2_coefs["boot_ci95_high"].to_numpy()).all()
    assert (m2_coefs["boot_ci95_low"] <= m2_coefs["estimate_reml"]).all()
    assert (m2_coefs["estimate_reml"] <= m2_coefs["boot_ci95_high"]).all()
    assert np.isfinite(
        m2_coefs[m2_coefs["term"].str.contains("sim_")]["fdr_bh_q_value"].to_numpy()
    ).all()

    h2_df = pd.read_csv(h2_path)
    assert {"in-domain", "partial-overlap", "out-of-domain"}.issubset(set(h2_df["stratum"]))

    cv_df = pd.read_csv(cv_path)
    req_preds = {"Constant", "TanJerome_Simon_Heuristic", "M1", "M2", "M3"}
    assert req_preds.issubset(set(cv_df["predictor"]))
    assert np.isfinite(cv_df["oos_rmse"].to_numpy()).all()
    assert np.isfinite(cv_df["oos_mae"].to_numpy()).all()
    assert np.isfinite(cv_df["sign_accuracy"].to_numpy()).all()
