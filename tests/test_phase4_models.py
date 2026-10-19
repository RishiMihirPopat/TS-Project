"""Phase 4 Acceptance Tests: Zero-Shot TSFM Inference & Compute Profiling."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tsdomain.baselines.run_baselines import LOCAL_CONFIGS
from tsdomain.utils import PROJECT_ROOT, validate_forecast_dataframe

P0_TSFM_MODELS = ["Chronos_small", "Moirai_small", "TimesFM"]
ALL_LOCAL_TSFMS = ["Chronos_small", "Moirai_small", "TimesFM", "TTM-R2-Pretrained"]


def test_phase4_forecast_parquets_and_metadata() -> None:
    """Verify forecast parquets and sidecar .meta.json files exist and pass schema checks."""
    forecasts_root = PROJECT_ROOT / "results" / "forecasts"
    for model_id in ALL_LOCAL_TSFMS:
        model_dir = forecasts_root / model_id
        assert model_dir.exists(), f"Missing forecast directory for {model_id}"
        for cfg in LOCAL_CONFIGS:
            safe_cfg = cfg.replace("/", "__")
            pq_path = model_dir / f"{safe_cfg}.parquet"
            meta_path = model_dir / f"{safe_cfg}.parquet.meta.json"
            assert pq_path.exists(), f"Missing forecast parquet {pq_path}"
            assert meta_path.exists(), f"Missing metadata sidecar {meta_path}"

            df = pd.read_parquet(pq_path)
            validate_forecast_dataframe(df)
            assert np.isfinite(df["mean"].to_numpy()).all(), f"Non-finite mean in {pq_path}"
            for qcol in ["q0.1", "q0.5", "q0.9"]:
                assert np.isfinite(df[qcol].to_numpy()).all(), f"Non-finite {qcol} in {pq_path}"

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert meta["model_id"] == model_id
            assert meta["runtime_s"] > 0.0
            assert meta["peak_ram_mb"] > 0.0


def test_phase4_compute_profile_and_metrics() -> None:
    """Verify compute_profile.parquet and tsfm_local_metrics.parquet have all required columns and models."""
    comp_path = PROJECT_ROOT / "results" / "compute" / "compute_profile.parquet"
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "tsfm_local_metrics.parquet"
    assert comp_path.exists(), "Missing results/compute/compute_profile.parquet"
    assert metrics_path.exists(), "Missing results/metrics/tsfm_local_metrics.parquet"

    comp_df = pd.read_parquet(comp_path)
    req_comp_cols = [
        "model",
        "config",
        "param_count",
        "device",
        "load_time_s",
        "infer_time_s_total",
        "infer_time_ms_per_point",
        "infer_time_iqr_ms",
        "peak_ram_mb",
        "peak_vram_mb",
        "energy_wh_total",
        "energy_mwh_per_1k_points",
        "energy_source",
    ]
    for col in req_comp_cols:
        assert col in comp_df.columns, f"Missing column {col} in compute_profile.parquet"
        assert not comp_df[col].isna().any(), f"Null values in compute_profile.{col}"

    for m in ALL_LOCAL_TSFMS:
        sub = comp_df[comp_df["model"] == m]
        assert len(sub) == len(LOCAL_CONFIGS), f"Expected {len(LOCAL_CONFIGS)} rows for {m}, got {len(sub)}"
        assert (sub["infer_time_ms_per_point"] > 0).all()
        assert (sub["energy_mwh_per_1k_points"] > 0).all()
        assert (sub["peak_ram_mb"] > 0).all()

    metrics_df = pd.read_parquet(metrics_path)
    assert len(metrics_df) == len(ALL_LOCAL_TSFMS) * len(LOCAL_CONFIGS)
    assert np.isfinite(metrics_df["mase"].to_numpy()).all()
    assert np.isfinite(metrics_df["crps"].to_numpy()).all()
