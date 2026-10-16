"""Phase 3: Categorical and Continuous Pretraining Domain-Similarity Scoring and VIF Diagnostics.

Implements:
1. Categorical `domain_in_pretraining` (`true`, `false`, `unknown`) from `pretraining_registry.yaml`,
   plus sensitivity variants (`domain_in_pretraining_optimistic` where `unknown -> True`, and
   `domain_in_pretraining_conservative` where `unknown -> False`).
2. Continuous `sim_distance`: regularized Mahalanobis distance (and k-NN Euclidean distance)
   from a test dataset's standardized meta-feature vector to the pretraining reference pool
   for each model (leaving out the test dataset's own `base_dataset` to prevent self-match).
3. Pairwise correlation matrix and Variance Inflation Factor (VIF) table across meta-features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tsdomain.config import PretrainingRegistrySchema

SIMILARITY_FEATURE_COLS: List[str] = [
    "trend_strength",
    "seasonal_strength",
    "forecastability",
    "hurst",
    "stability",
    "lumpiness",
    "log_length",
]


def compute_vif_and_correlations(
    features_df: pd.DataFrame,
    feature_cols: List[str] = SIMILARITY_FEATURE_COLS,
    out_dir: Path = Path("results/features"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Pearson/Spearman correlations and Variance Inflation Factors (VIF)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_df = features_df[feature_cols].copy().dropna()

    corr_df = X_df.corr(method="pearson")
    corr_df.to_csv(out_dir / "feature_correlations.csv")

    # Standardize columns and compute VIF_j = 1 / (1 - R_j^2)
    X = X_df.to_numpy(dtype=float)
    X_std = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0, ddof=1), 1e-12)
    corr_mat = np.corrcoef(X_std, rowvar=False)
    # Regularize slightly in case of near-collinearity
    inv_corr = np.linalg.pinv(corr_mat + 1e-8 * np.eye(len(feature_cols)))
    vifs = np.diag(inv_corr)

    vif_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "vif": [float(max(1.0, v)) for v in vifs],
            "keep_under_vif_10": [bool(v < 10.0) for v in vifs],
        }
    )
    vif_df.to_csv(out_dir / "vif_report.csv", index=False)
    return corr_df, vif_df


def compute_model_dataset_similarity(
    features_df: pd.DataFrame,
    registry: PretrainingRegistrySchema,
    feature_cols: List[str] = SIMILARITY_FEATURE_COLS,
    k_neighbors: int = 3,
    out_parquet: Path = Path("results/features/model_dataset_similarity.parquet"),
) -> pd.DataFrame:
    """Compute categorical and continuous domain similarity for all (model, dataset_config) pairs."""
    df_f = features_df.copy().reset_index(drop=True)
    X_raw = df_f[feature_cols].to_numpy(dtype=float)

    # Robust z-score standardization across the benchmark pool
    mu_global = np.mean(X_raw, axis=0)
    sd_global = np.maximum(np.std(X_raw, axis=0, ddof=1), 1e-8)
    Z = (X_raw - mu_global) / sd_global

    rows: List[Dict] = []

    for model_name, model_entry in registry.models.items():
        dom_map = model_entry.domains_covered or {}

        # Determine categorical exposure for every config in df_f
        cat_labels: List[str] = []
        verified_flags: List[bool] = []
        for _, r in df_f.iterrows():
            dom = str(r["domain"])
            if dom in dom_map:
                cat_labels.append(dom_map[dom].in_pretraining)
                verified_flags.append(bool(dom_map[dom].verified))
            else:
                cat_labels.append("false")
                verified_flags.append(False)

        # Reference pool indices: datasets documented as in_pretraining == "true"
        ref_mask = np.array([lbl == "true" for lbl in cat_labels], dtype=bool)
        if np.sum(ref_mask) < 3:
            ref_mask = np.ones(len(df_f), dtype=bool)

        for i, r in df_f.iterrows():
            cfg = str(r["config"])
            base_ds = str(r["base_dataset"])
            dom = str(r["domain"])
            cat = cat_labels[i]
            ver = verified_flags[i]

            # Leave-one-base-dataset-out reference pool so a dataset is not compared to itself
            loo_mask = ref_mask & (df_f["base_dataset"].to_numpy() != base_ds)
            if np.sum(loo_mask) < 3:
                loo_mask = df_f["base_dataset"].to_numpy() != base_ds

            Z_ref = Z[loo_mask]
            z_i = Z[i]

            # 1. Regularized Mahalanobis distance to reference pool centroid
            mu_ref = np.mean(Z_ref, axis=0)
            cov_ref = np.cov(Z_ref, rowvar=False) + 0.25 * np.eye(len(feature_cols))
            inv_cov = np.linalg.pinv(cov_ref)
            diff = z_i - mu_ref
            mahal_dist = float(np.sqrt(max(0.0, float(diff.T @ inv_cov @ diff))))

            # 2. Mean Euclidean distance to k nearest reference datasets (unique by base_dataset)
            dists_all = np.linalg.norm(Z_ref - z_i, axis=1)
            ref_bases = df_f.loc[loo_mask, "base_dataset"].to_numpy()
            base_min_dists = []
            for b in np.unique(ref_bases):
                base_min_dists.append(float(np.min(dists_all[ref_bases == b])))
            base_min_dists.sort()
            k_eff = min(k_neighbors, len(base_min_dists))
            knn_dist = float(np.mean(base_min_dists[:k_eff])) if k_eff > 0 else mahal_dist

            rows.append(
                {
                    "model": model_name,
                    "config": cfg,
                    "base_dataset": base_ds,
                    "domain": dom,
                    "domain_in_pretraining": cat,
                    "domain_in_pretraining_optimistic": bool(cat in {"true", "unknown"}),
                    "domain_in_pretraining_conservative": bool(cat == "true"),
                    "claim_verified": ver,
                    "finance_included": bool(model_entry.finance_included),
                    "training_cutoff": str(model_entry.training_cutoff),
                    "sim_distance": mahal_dist,
                    "sim_distance_knn": knn_dist,
                }
            )

    sim_df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_parquet(out_parquet, index=False)
    sim_df.to_csv(out_parquet.with_suffix(".csv"), index=False)
    return sim_df
