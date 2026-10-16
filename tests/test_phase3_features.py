"""Phase 3 acceptance tests: synthetic meta-feature sanity (sine vs noise), VIF, and similarity table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsdomain.features.meta_features import compute_series_meta_features


def test_synthetic_sine_vs_white_noise_meta_features() -> None:
    """Acceptance check: pure sine wave has high seasonal strength and low spectral entropy; white noise the opposite."""
    t = np.arange(240, dtype=float)
    sine_wave = np.sin(2.0 * np.pi * t / 12.0)

    rng = np.random.default_rng(12345)
    white_noise = rng.normal(0.0, 1.0, size=240)

    f_sine = compute_series_meta_features(sine_wave, stl_period=12)
    f_noise = compute_series_meta_features(white_noise, stl_period=12)

    assert f_sine["seasonal_strength"] > 0.90, f"Expected high seasonal_strength on sine, got {f_sine['seasonal_strength']}"
    assert f_sine["spectral_entropy"] < 0.35, f"Expected low spectral_entropy on sine, got {f_sine['spectral_entropy']}"
    assert f_sine["forecastability"] > 0.65

    assert f_noise["seasonal_strength"] < 0.25, f"Expected low seasonal_strength on white noise, got {f_noise['seasonal_strength']}"
    assert f_noise["spectral_entropy"] > 0.80, f"Expected high spectral_entropy on white noise, got {f_noise['spectral_entropy']}"
    assert f_noise["forecastability"] < 0.20


def test_dataset_features_and_similarity_outputs() -> None:
    """Acceptance check: dataset_features.parquet, model_dataset_similarity.parquet, corr, and VIF exist."""
    feat_path = Path("results/features/dataset_features.parquet")
    sim_path = Path("results/features/model_dataset_similarity.parquet")
    corr_path = Path("results/features/feature_correlations.csv")
    vif_path = Path("results/features/vif_report.csv")

    assert feat_path.exists()
    assert sim_path.exists()
    assert corr_path.exists()
    assert vif_path.exists()

    df_feat = pd.read_parquet(feat_path)
    df_sim = pd.read_parquet(sim_path)
    df_vif = pd.read_csv(vif_path)

    # All 97 GIFT-Eval configs + 2 unseen configs = 99
    assert len(df_feat) == 99
    assert "unseen_mandi/W/short" in set(df_feat["config"])
    assert "unseen_recency/D/short" in set(df_feat["config"])

    for col in [
        "trend_strength",
        "seasonal_strength",
        "spectral_entropy",
        "forecastability",
        "hurst",
        "stability",
        "lumpiness",
        "length",
        "log_length",
    ]:
        assert not df_feat[col].isna().any(), f"NaN in dataset_features[{col}]"

    assert not df_sim["sim_distance"].isna().any()
    assert set(df_sim["domain_in_pretraining"].unique()).issubset({"true", "false", "unknown"})
    assert len(df_vif) == 7
