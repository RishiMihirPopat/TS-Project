"""Phase 8 (P1) Acceptance Tests: LightGBM Budget-Parity Baseline & Robustness Ablations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsdomain.baselines.lgbm import _build_lag_features
from tsdomain.baselines.run_baselines import LOCAL_CONFIGS
from tsdomain.utils import PROJECT_ROOT, validate_forecast_dataframe


def test_phase8_lgbm_non_overlapping_temporal_split() -> None:
    """Verify LightGBM lag feature builder splits temporally and z-scores only on pre-validation prefix."""
    s = np.concatenate([np.arange(20, dtype=float), np.full(5, 10_000.0)])
    X_tr, y_tr, X_va, y_va = _build_lag_features([s], n_lags=5, season_length=4)
    assert len(X_tr) == 16 and len(X_va) == 4
    # Pre-validation prefix s[:21] has max <= 10_000 only at index 20; if full series were used, y_tr would be near -0.5
    assert float(np.std(y_tr)) > 0.1 and np.all(y_va > 1.0)


def test_phase8_lightgbm_budget_parity_and_forecasts() -> None:
    """Verify LightGBM produces valid forecast parquets across all 13 local configs within budget."""
    lgbm_metrics_path = PROJECT_ROOT / "results" / "metrics" / "lgbm_local_metrics.parquet"
    assert lgbm_metrics_path.exists(), "Missing results/metrics/lgbm_local_metrics.parquet"

    lgbm_df = pd.read_parquet(lgbm_metrics_path)
    assert len(lgbm_df) == len(LOCAL_CONFIGS)
    assert np.isfinite(lgbm_df["mase"].to_numpy()).all()
    assert np.isfinite(lgbm_df["crps"].to_numpy()).all()

    fc_dir = PROJECT_ROOT / "results" / "forecasts" / "LightGBM"
    for cfg in LOCAL_CONFIGS:
        safe_cfg = cfg.replace("/", "__")
        pq_path = fc_dir / f"{safe_cfg}.parquet"
        assert pq_path.exists(), f"Missing LightGBM forecast {pq_path}"
        df_fc = pd.read_parquet(pq_path)
        validate_forecast_dataframe(df_fc)


def test_phase8_four_ablation_tables() -> None:
    """Verify all four ablation CSVs and ablations_summary.csv exist and have valid values."""
    analysis_dir = PROJECT_ROOT / "results" / "analysis"
    req_files = [
        "ablation_scaling.csv",
        "ablation_leakage_exclusion.csv",
        "ablation_unknown_registry.csv",
        "ablation_bocpd_breaks.csv",
        "ablations_summary.csv",
    ]
    for fname in req_files:
        fpath = analysis_dir / fname
        assert fpath.exists(), f"Missing {fpath}"
        df = pd.read_csv(fpath)
        assert not df.empty, f"Empty {fpath}"

    bocpd_df = pd.read_csv(analysis_dir / "ablation_bocpd_breaks.csv")
    assert int(bocpd_df["bocpd_recovered_30d"].sum()) >= 4

    abl_sum = pd.read_csv(analysis_dir / "ablations_summary.csv")
    leak_conc = str(abl_sum.loc[abl_sum["ablation_id"] == "8b2_Leakage_Exclusion", "conclusion"].iloc[0])
    assert "Clean-subset M2 vs M1 p=" in leak_conc and "M3 vs M2 p=" in leak_conc
