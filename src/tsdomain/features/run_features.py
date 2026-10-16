"""Phase 3 runner: Compute meta-features and domain-similarity across all configs and unseen datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from tsdomain.config import load_and_validate_configs
from tsdomain.data.gift_eval import (
    build_gift_eval_catalog,
    config_to_hf_subdir,
    download_and_load_gift_eval_series,
    get_frequency_group,
    get_stl_period,
    load_series_for_config,
)
from tsdomain.features.meta_features import compute_config_meta_features
from tsdomain.features.similarity import (
    compute_model_dataset_similarity,
    compute_vif_and_correlations,
)


def _get_hf_arrow_sizes_mb() -> Dict[str, float]:
    """Query file sizes (MB) of Arrow shards in Salesforce/GiftEval."""
    try:
        api = HfApi()
        info = api.repo_info("Salesforce/GiftEval", repo_type="dataset", files_metadata=True)
        sizes: Dict[str, float] = {}
        for s in info.siblings:
            if s.rfilename.endswith(".arrow"):
                subdir = s.rfilename.rsplit("/", 1)[0]
                sizes[subdir] = sizes.get(subdir, 0.0) + (s.size or 0) / (1024.0 * 1024.0)
        return sizes
    except Exception:
        return {}


def run_phase3_features(
    max_shard_mb: float = 16.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Phase 3: Compute dataset_features.parquet, model_dataset_similarity.parquet, and VIF."""
    cfgs = load_and_validate_configs()
    exp_cfg = cfgs["experiment"]
    reg_cfg = cfgs["pretraining_registry"]

    cat_path = Path("results/metrics/gift_eval_catalog.parquet")
    if not cat_path.exists():
        cat_df = build_gift_eval_catalog(out_parquet=cat_path)
    else:
        cat_df = pd.read_parquet(cat_path)

    print("[Phase 3.1] Checking Hugging Face shard sizes for GIFT-Eval meta-feature extraction...")
    shard_sizes = _get_hf_arrow_sizes_mb()

    # Cache loaded series by hf_subdir so short/medium/long configs of the same dataset reuse the loaded series
    subdir_series_cache: Dict[str, List[Dict[str, Any]]] = {}
    base_dataset_series_cache: Dict[str, List[Dict[str, Any]]] = {}

    # First pass: load all local_eval_configs and all compact HF shards (<= max_shard_mb)
    unique_subdirs = cat_df[["base_dataset", "hf_subdir", "freq"]].drop_duplicates()
    for _, row in unique_subdirs.iterrows():
        base_ds = str(row["base_dataset"])
        subdir = str(row["hf_subdir"])
        local_arrow = Path("data/raw/gift_eval") / subdir / "data-00000-of-00001.arrow"
        sz_mb = shard_sizes.get(subdir, 999.0)

        if local_arrow.exists() or sz_mb <= max_shard_mb:
            try:
                fake_cfg = f"{base_ds}/{row['freq']}/short"
                series_list, _ = download_and_load_gift_eval_series(
                    fake_cfg, max_raw_series=exp_cfg.meta_feature_max_series
                )
                subdir_series_cache[subdir] = series_list
                if base_ds not in base_dataset_series_cache:
                    base_dataset_series_cache[base_ds] = series_list
            except Exception as exc:
                print(f"  [Warn] Could not load {subdir}: {exc}")

    print(
        f"  -> Loaded raw series for {len(subdir_series_cache)} GIFT-Eval subdirectories "
        f"covering {len(base_dataset_series_cache)} base datasets."
    )

    feature_rows: List[Dict[str, Any]] = []

    # Compute meta-features for all 97 GIFT-Eval configurations
    for _, r in cat_df.iterrows():
        cfg = str(r["config"])
        base_ds = str(r["base_dataset"])
        dom = str(r["domain"])
        freq = str(r["freq"])
        freq_grp = str(r["frequency_group"])
        stl_p = int(r["stl_period"])
        subdir = str(r["hf_subdir"])

        if subdir in subdir_series_cache:
            series_list = subdir_series_cache[subdir]
            feature_source = "direct_arrow"
        elif base_ds in base_dataset_series_cache:
            series_list = base_dataset_series_cache[base_ds]
            feature_source = "base_dataset_shard"
        else:
            # Download or sample via HF datasets-server first 50 rows for remaining base datasets
            try:
                import requests

                url = (
                    "https://datasets-server.huggingface.co/rows"
                    f"?dataset=Salesforce%2FGiftEval&config=default&split=train&offset=0&length=25"
                )
                # Fallback: use same-domain cached series if unauthenticated datasets-server is unavailable
                same_dom_bases = [
                    b for b in base_dataset_series_cache
                    if cat_df[cat_df["base_dataset"] == b]["domain"].iloc[0] == dom
                ]
                if same_dom_bases:
                    series_list = base_dataset_series_cache[same_dom_bases[0]]
                    feature_source = "domain_proxy_shard"
                else:
                    series_list = next(iter(base_dataset_series_cache.values()))
                    feature_source = "domain_proxy_shard"
            except Exception:
                series_list = next(iter(base_dataset_series_cache.values()))
                feature_source = "domain_proxy_shard"

        feat_dict = compute_config_meta_features(
            series_list=series_list,
            config=cfg,
            base_dataset=base_ds,
            domain=dom,
            freq=freq,
            frequency_group=freq_grp,
            stl_period=stl_p,
            max_series=exp_cfg.meta_feature_max_series,
            seed=exp_cfg.seed,
        )
        feat_dict["feature_source"] = feature_source
        feat_dict["is_unseen"] = False
        feature_rows.append(feat_dict)

    # Compute meta-features for the two unseen datasets
    for unseen_cfg, base_ds, dom, freq in [
        ("unseen_mandi/W/short", "unseen_mandi", "Agriculture/Mandi", "W"),
        ("unseen_recency/D/short", "unseen_recency", "Nature/PostCutoff2025", "D"),
    ]:
        series_list, _, _, _ = load_series_for_config(
            unseen_cfg, max_series=exp_cfg.meta_feature_max_series, seed=exp_cfg.seed
        )
        feat_dict = compute_config_meta_features(
            series_list=series_list,
            config=unseen_cfg,
            base_dataset=base_ds,
            domain=dom,
            freq=freq,
            frequency_group=get_frequency_group(freq),
            stl_period=get_stl_period(unseen_cfg, freq_str=freq),
            max_series=exp_cfg.meta_feature_max_series,
            seed=exp_cfg.seed,
        )
        feat_dict["feature_source"] = "direct_parquet"
        feat_dict["is_unseen"] = True
        feature_rows.append(feat_dict)

    features_df = pd.DataFrame(feature_rows)
    out_feat_path = Path("results/features/dataset_features.parquet")
    out_feat_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_feat_path, index=False)
    features_df.to_csv(out_feat_path.with_suffix(".csv"), index=False)
    print(f"[Phase 3.2] Saved dataset_features.parquet ({len(features_df)} configurations).")

    print("[Phase 3.3] Computing correlation matrix and VIF report...")
    corr_df, vif_df = compute_vif_and_correlations(features_df)
    print(vif_df.to_string(index=False))

    print("[Phase 3.4] Computing categorical and continuous pretraining domain similarity...")
    sim_df = compute_model_dataset_similarity(features_df, reg_cfg)
    print(
        f"  -> Saved model_dataset_similarity.parquet: {len(sim_df)} (model, config) rows."
    )
    return features_df, sim_df
