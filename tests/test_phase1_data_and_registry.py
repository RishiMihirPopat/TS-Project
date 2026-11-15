"""Phase 1 acceptance tests: YAML schemas, GIFT-Eval leaderboard, catalog, and unseen dataset loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsdomain.config import load_and_validate_configs


def test_configs_validate_against_schema() -> None:
    """Acceptance check: all 4 YAML configs validate against Pydantic schemas."""
    cfgs = load_and_validate_configs()
    assert len(cfgs["datasets"].datasets) >= 12
    assert len(cfgs["models"].models) >= 9
    assert "Chronos_small" in cfgs["pretraining_registry"].models
    assert "Moirai_small" in cfgs["pretraining_registry"].models
    assert "TimesFM" in cfgs["pretraining_registry"].models

    # Verify every claim has source_url and verified boolean
    for model_name, entry in cfgs["pretraining_registry"].models.items():
        assert entry.domains_covered is not None, f"Missing domains_covered on {model_name}"
        for dom, claim in entry.domains_covered.items():
            assert claim.in_pretraining in {"true", "false", "unknown", "partial"}
            assert isinstance(claim.verified, bool)
            assert claim.source_url.startswith("http")


def test_leaderboard_table_integrity() -> None:
    """Acceptance check: leaderboard.parquet loads with expected models and zero duplicate keys."""
    lb_path = Path("results/metrics/leaderboard.parquet")
    assert lb_path.exists(), "results/metrics/leaderboard.parquet missing; run `make data` first."
    df = pd.read_parquet(lb_path)

    required_cols = {
        "model",
        "model_type",
        "config",
        "base_dataset",
        "domain",
        "mase",
        "crps",
        "test_data_leakage_flag",
    }
    assert required_cols.issubset(set(df.columns))

    # No duplicate (model, config) keys
    assert not df.duplicated(subset=["model", "config"]).any()

    # Check required model families are present
    models_present = set(df["model"].unique())
    for req in [
        "Naive",
        "Seasonal_Naive",
        "Auto_Arima",
        "Auto_ETS",
        "Auto_Theta",
        "PatchTST",
        "N-BEATS",
        "Chronos_small",
        "Moirai_small",
        "TimesFM",
        "TTM-R2-Pretrained",
    ]:
        assert req in models_present, f"Required model {req} missing from leaderboard.parquet"

    # 97 GIFT-Eval configurations per model
    sn_count = len(df[df["model"] == "Seasonal_Naive"])
    assert sn_count == 97


def test_unseen_datasets_loader_invariants() -> None:
    """Acceptance check: both unseen datasets have >= 50 series, no duplicate timestamps, regular freq."""
    for name, path_str, min_pts in [
        ("unseen_mandi", "data/processed/unseen_mandi.parquet", 200),
        ("unseen_recency", "data/processed/unseen_recency.parquet", 500),
    ]:
        p = Path(path_str)
        assert p.exists(), f"{path_str} does not exist"
        df = pd.read_parquet(p)
        assert {"item_id", "timestamp", "target"}.issubset(set(df.columns))
        assert not df["target"].isna().any(), f"NaNs found in {name}"
        n_series = df["item_id"].nunique()
        assert n_series >= 50, f"Expected >= 50 series in {name}, got {n_series}"
        assert not df.duplicated(subset=["item_id", "timestamp"]).any(), f"Duplicate timestamps in {name}"

        # Check series length and regular frequency on first series
        first_id = df["item_id"].iloc[0]
        sub = df[df["item_id"] == first_id].sort_values("timestamp")
        assert len(sub) >= min_pts
        diffs = sub["timestamp"].diff().dropna()
        assert diffs.nunique() == 1, f"Irregular timestamp spacing in {name}: {diffs.unique()}"
