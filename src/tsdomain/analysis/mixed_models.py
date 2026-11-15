"""Phase 6: Hierarchical Mixed-Effects Models for Predicting Zero-Shot TSFM Gains (H1 & H2)."""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from tsdomain.config import load_and_validate_configs
from tsdomain.utils import PROJECT_ROOT, ensure_dir

CHAR_COLS = [
    "forecastability",
    "seasonal_strength",
    "trend_strength",
    "hurst",
    "stability",
    "lumpiness",
    "log_length",
]

SIM_COLS = [
    "sim_binary",
    "sim_share",
    "sim_distance",
]


def build_phase6_analysis_dataset() -> pd.DataFrame:
    """Assemble the (model, config) analysis table combining GIFT-Eval leaderboard and local runs."""
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    features_dir = PROJECT_ROOT / "results" / "features"

    lb_df = pd.read_parquet(metrics_dir / "leaderboard.parquet")
    best_cls_lb = pd.read_parquet(metrics_dir / "best_classical_leaderboard.parquet")
    feat_df = pd.read_parquet(features_dir / "dataset_features.parquet")
    sim_df = pd.read_parquet(features_dir / "model_dataset_similarity.parquet")

    # 1. GIFT-Eval 97 configs x 15 TSFMs from leaderboard
    tsfm_lb = lb_df[lb_df["model_type"].isin(["pretrained", "zero-shot", "tsfm"]) | lb_df["model"].isin(sim_df["model"].unique())].copy()
    tsfm_lb = tsfm_lb[tsfm_lb["model"].isin(sim_df["model"].unique())].copy()

    merged_lb = tsfm_lb.merge(
        best_cls_lb[["config", "best_classical_model", "best_classical_mase", "best_classical_crps"]],
        on="config",
        how="inner",
    )
    merged_lb["source"] = "gift_eval_leaderboard"

    # 2. Unseen configs (unseen_mandi/W/short, unseen_recency/D/short) from local Phase 4 TSFM runs
    local_tsfm_path = metrics_dir / "tsfm_local_metrics.parquet"
    best_cls_local_path = metrics_dir / "best_classical_local.parquet"

    rows_list = [merged_lb]
    if local_tsfm_path.exists() and best_cls_local_path.exists():
        local_tsfm = pd.read_parquet(local_tsfm_path)
        best_cls_loc = pd.read_parquet(best_cls_local_path)
        unseen_tsfm = local_tsfm[local_tsfm["config"].str.startswith("unseen_")].copy()
        if not unseen_tsfm.empty:
            unseen_merged = unseen_tsfm.merge(
                best_cls_loc[["config", "base_dataset", "term", "best_classical_model", "best_classical_mase", "best_classical_crps"]],
                on="config",
                how="inner",
            )
            unseen_merged["source"] = "local_unseen"
            rows_list.append(unseen_merged)

    combined = pd.concat(rows_list, ignore_index=True)

    # Join dataset meta-features
    feat_keep = [
        "config",
        "frequency_group",
        "forecastability",
        "seasonal_strength",
        "trend_strength",
        "spectral_entropy",
        "hurst",
        "stability",
        "lumpiness",
        "length",
        "log_length",
    ]
    combined = combined.merge(feat_df[feat_keep], on="config", how="inner")

    # Join domain similarity & model metadata from similarity.parquet
    sim_keep = [
        "model",
        "config",
        "domain_in_pretraining",
        "domain_in_pretraining_optimistic",
        "domain_in_pretraining_conservative",
        "dataset_leak",
        "claim_verified",
        "finance_included",
        "training_cutoff",
        "sim_distance",
        "sim_distance_knn",
    ]
    combined = combined.merge(
        sim_df[[c for c in sim_keep if c in sim_df.columns]],
        on=["model", "config"],
        how="inner",
    )

    reg_models = load_and_validate_configs()["pretraining_registry"].models
    domain_share_by_family = {
        "chronos_t5": {"Energy": 0.29, "Transport": 0.24, "Nature": 0.20, "Web/CloudOps": 0.13, "Sales": 0.12, "Healthcare": 0.0, "Econ/Fin": 0.02},
        "chronos_bolt": {"Energy": 0.29, "Transport": 0.24, "Nature": 0.20, "Web/CloudOps": 0.13, "Sales": 0.12, "Healthcare": 0.0, "Econ/Fin": 0.02},
        "chronos_2": {"Energy": 0.26, "Transport": 0.22, "Nature": 0.20, "Web/CloudOps": 0.14, "Sales": 0.12, "Econ/Fin": 0.04, "Healthcare": 0.02},
        "moirai_1": {"Energy": 0.34, "Nature": 0.28, "Transport": 0.18, "Web/CloudOps": 0.10, "Sales": 0.05, "Econ/Fin": 0.03, "Healthcare": 0.02},
        "moirai_2": {"Energy": 0.32, "Nature": 0.26, "Transport": 0.18, "Web/CloudOps": 0.12, "Sales": 0.06, "Econ/Fin": 0.04, "Healthcare": 0.02},
        "timesfm": {"Web/CloudOps": 0.32, "Nature": 0.20, "Energy": 0.18, "Transport": 0.16, "Sales": 0.12, "Econ/Fin": 0.01, "Healthcare": 0.01},
        "ttm": {"Energy": 0.38, "Transport": 0.25, "Nature": 0.25, "Web/CloudOps": 0.08, "Sales": 0.04, "Econ/Fin": 0.0, "Healthcare": 0.0},
    }

    combined["model_family"] = combined["model"].map(
        lambda m: str(reg_models[str(m)].family)
        if str(m) in reg_models and reg_models[str(m)].family
        else str(m).split("_")[0].split("-")[0].lower()
    )
    combined["param_count"] = combined["model"].map(
        lambda m: float(reg_models[str(m)].param_count)
        if str(m) in reg_models and reg_models[str(m)].param_count
        else 50_000_000.0
    )
    combined["sim_binary"] = combined["domain_in_pretraining"].map(
        lambda x: 1.0 if str(x) == "true" else (0.5 if str(x) == "partial" else 0.0)
    )
    combined["sim_binary_optimistic"] = combined["domain_in_pretraining_optimistic"].astype(float)

    def _compute_share(row: pd.Series) -> float:
        dip = str(row["domain_in_pretraining"])
        if dip == "false":
            return 0.0
        fam = str(row["model_family"])
        dom = str(row["domain"])
        sh = domain_share_by_family.get(fam, {}).get(dom, 0.05)
        if dip in ("unknown", "partial"):
            return 0.5 * sh
        return float(sh)

    def _compute_stratum(row: pd.Series) -> str:
        cfg = str(row["config"])
        dip = str(row["domain_in_pretraining"])
        if cfg.startswith("unseen_recency"):
            return "unknown_recency" if dip == "unknown" else "recency"
        if cfg.startswith("unseen_mandi"):
            return "unseen_dataset_overlapping_domain"
        if dip == "false":
            return "out-of-domain"
        if dip in ("unknown", "partial") or float(row["sim_share"]) < 0.08:
            return "partial-overlap"
        return "in-domain"

    combined["sim_share"] = combined.apply(_compute_share, axis=1)
    combined["sim_stratum"] = combined.apply(_compute_stratum, axis=1)
    leak_bases = {"ett1", "ett2", "solar", "saugeen", "kdd_cup_2018", "electricity"}
    leak_flag = (
        combined["test_data_leakage_flag"].eq(True)
        if "test_data_leakage_flag" in combined.columns
        else True
    )
    reg_leak = (
        combined["dataset_leak"].astype(bool)
        if "dataset_leak" in combined.columns
        else combined["base_dataset"].isin(leak_bases)
    )
    combined["dataset_leak"] = (
        (leak_flag & reg_leak) | (combined["base_dataset"].isin(leak_bases) & leak_flag)
    ).astype(int)
    combined["in_pretraining_pool"] = combined["dataset_leak"]
    combined["source_local"] = (combined["source"] == "local_unseen").astype(int)

    # Response variable: y = log(MASE_model / MASE_classical); report (1 - exp(y)) * 100 as % gain
    combined["mase_ratio"] = (
        combined["mase"] / np.maximum(combined["best_classical_mase"], 1e-6)
    ).clip(lower=0.05, upper=10.0)
    combined["y_log_mase_ratio"] = np.log(combined["mase_ratio"])
    combined["pct_gain"] = (1.0 - np.exp(combined["y_log_mase_ratio"])) * 100.0
    combined["delta_mase_raw"] = (1.0 - combined["mase"] / np.maximum(combined["best_classical_mase"], 1e-6))
    # Orient delta_mase = -y_log_mase_ratio = log(MASE_classical / MASE_model) so positive = TSFM improvement
    combined["delta_mase"] = -combined["y_log_mase_ratio"]
    combined["log_mase_ratio"] = combined["delta_mase"]
    combined["delta_crps"] = -np.log(
        (combined["crps"] / np.maximum(combined["best_classical_crps"], 1e-6)).clip(lower=0.05, upper=10.0)
    )
    combined["tsfm_wins"] = (combined["y_log_mase_ratio"] < 0.0).astype(int)
    combined["log_params"] = np.log10(np.maximum(combined["param_count"].astype(float), 1.0))

    # Standardize continuous predictors (mean 0, SD 1)
    to_standardize = CHAR_COLS + ["sim_share", "sim_distance", "sim_distance_knn", "log_params"]
    for col in to_standardize:
        vals = combined[col].to_numpy(dtype=float)
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1))
        if sd < 1e-12:
            sd = 1.0
        combined[f"z_{col}"] = (vals - mu) / sd

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "analysis")
    combined.to_parquet(out_dir / "phase6_analysis_dataset.parquet", index=False)
    return combined


def _fit_crossed_mixedlm(
    formula: str,
    df: pd.DataFrame,
    reml: bool = False,
):
    """Fit MixedLM with random intercepts (1|base_dataset) + (1|config) + (1|model_family)."""
    df_fit = df.copy()
    df_fit["_all_group"] = 1
    vc = {
        "base_dataset": "0 + C(base_dataset)",
        "config": "0 + C(config)",
        "model_family": "0 + C(model_family)",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            formula,
            data=df_fit,
            groups=df_fit["_all_group"],
            vc_formula=vc,
        )
        try:
            res = model.fit(reml=reml, method="lbfgs", maxiter=500)
        except Exception:
            res = model.fit(reml=reml, method="powell")
    return res


def compute_nakagawa_r2(res, df: pd.DataFrame) -> Tuple[float, float, Dict[str, float]]:
    """Compute Nakagawa & Schielzeth marginal R^2, conditional R^2, and ICC breakdown."""
    fe_params = res.fe_params
    exog_mat = res.model.exog
    fe_pred = np.dot(exog_mat, fe_params.to_numpy())
    var_fixed = float(np.var(fe_pred, ddof=1)) if len(fe_pred) > 1 else 0.0

    # Random-effects variance components (vcomp in alphabetical order of vc keys: base_dataset, config, model_family)
    vcomp = np.asarray(res.vcomp, dtype=float)
    var_dataset = float(max(0.0, vcomp[0])) if len(vcomp) >= 1 else 0.0
    var_config = float(max(0.0, vcomp[1])) if len(vcomp) >= 3 else 0.0
    var_family = float(max(0.0, vcomp[-1])) if len(vcomp) >= 2 else 0.0
    var_resid = float(max(1e-12, res.scale))

    var_re = var_dataset + var_config + var_family
    var_total = var_fixed + var_re + var_resid
    r2_marginal = var_fixed / var_total
    r2_conditional = (var_fixed + var_re) / var_total
    icc_dataset = (var_dataset + var_config) / (var_re + var_resid)
    icc_config = var_config / (var_re + var_resid)
    icc_family = var_family / (var_re + var_resid)

    return (
        float(r2_marginal),
        float(r2_conditional),
        {
            "var_fixed": var_fixed,
            "var_base_dataset": var_dataset,
            "var_config": var_config,
            "var_model_family": var_family,
            "var_residual": var_resid,
            "icc_base_dataset": icc_dataset,
            "icc_config": icc_config,
            "icc_model_family": icc_family,
        },
    )


MODEL_FORMULAS: Dict[str, str] = {
    "M0": "delta_mase ~ 1",
    "M1": (
        "delta_mase ~ z_forecastability + z_seasonal_strength + z_trend_strength "
        "+ z_hurst + z_stability + z_lumpiness + z_log_length + z_log_params + dataset_leak + source_local + C(frequency_group)"
    ),
    "M2a": (
        "delta_mase ~ z_forecastability + z_seasonal_strength + z_trend_strength "
        "+ z_hurst + z_stability + z_lumpiness + z_log_length + z_log_params + dataset_leak + source_local + C(frequency_group) "
        "+ sim_binary + z_sim_share"
    ),
    "M2b": (
        "delta_mase ~ z_forecastability + z_seasonal_strength + z_trend_strength "
        "+ z_hurst + z_stability + z_lumpiness + z_log_length + z_log_params + dataset_leak + source_local + C(frequency_group) "
        "+ z_sim_distance"
    ),
    "M2": (
        "delta_mase ~ z_forecastability + z_seasonal_strength + z_trend_strength "
        "+ z_hurst + z_stability + z_lumpiness + z_log_length + z_log_params + dataset_leak + source_local + C(frequency_group) "
        "+ sim_binary + z_sim_share + z_sim_distance"
    ),
    "M3": (
        "delta_mase ~ z_forecastability + z_seasonal_strength + z_trend_strength "
        "+ z_hurst + z_stability + z_lumpiness + z_log_length + z_log_params + dataset_leak + source_local + C(frequency_group) "
        "+ sim_binary + z_sim_share + z_sim_distance "
        "+ z_sim_distance:z_forecastability + z_sim_distance:z_seasonal_strength + z_sim_distance:z_log_length"
    ),
}


def cluster_bootstrap_fixed_effects(
    formula: str,
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 20260410,
    res_reml=None,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% confidence intervals by resampling base_dataset clusters under MixedLM GLS (B = 1000)."""
    if res_reml is None:
        res_reml = _fit_crossed_mixedlm(formula, df, reml=True)

    fe_names = list(res_reml.fe_params.index)
    p = len(fe_names)
    X_full = np.asarray(res_reml.model.exog, dtype=float)
    y_full = np.asarray(res_reml.model.endog, dtype=float)
    Z_fam_full = pd.get_dummies(df["model_family"], dtype=float).to_numpy()
    n_fam = Z_fam_full.shape[1]

    vcomp = np.asarray(res_reml.vcomp, dtype=float)
    var_ds = max(float(vcomp[0]) + (float(vcomp[1]) if len(vcomp) >= 3 else 0.0), 1e-6)
    var_fam = max(float(vcomp[-1]) if len(vcomp) >= 2 else 0.01, 1e-6)
    var_res = max(float(res_reml.scale), 1e-6)

    rng = np.random.default_rng(seed)
    clusters = df["base_dataset"].unique()
    n_clusters = len(clusters)
    cluster_indices = {c: np.flatnonzero(df["base_dataset"].to_numpy() == c) for c in clusters}

    # Henderson's Mixed Model Equations solve exact MixedLM GLS per resampled cluster draw
    pen_diag = np.concatenate(
        [
            np.full(p, 1e-8),
            np.full(n_clusters, var_res / var_ds),
            np.full(n_fam, var_res / var_fam),
        ]
    )
    pen_mat = np.diag(pen_diag)
    boot_coefs = np.empty((n_boot, p), dtype=float)

    for b in range(n_boot):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        idx_list = [cluster_indices[c] for c in sampled]
        all_idx = np.concatenate(idx_list)
        X_b = X_full[all_idx]
        y_b = y_full[all_idx]
        Z_fam_b = Z_fam_full[all_idx]
        Z_ds_b = np.zeros((len(all_idx), n_clusters), dtype=float)
        pos = 0
        for j, idxs in enumerate(idx_list):
            m_len = len(idxs)
            Z_ds_b[pos : pos + m_len, j] = 1.0
            pos += m_len
        C = np.hstack([X_b, Z_ds_b, Z_fam_b])
        boot_coefs[b] = np.linalg.solve(C.T @ C + pen_mat, C.T @ y_b)[:p]

    return {
        k: (float(np.percentile(boot_coefs[:, i], 2.5)), float(np.percentile(boot_coefs[:, i], 97.5)))
        for i, k in enumerate(fe_names)
    }


def evaluate_h2_strata_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Test H2 across domain-overlap strata; strata with <5 base datasets are reported descriptively only."""
    strata_rows: List[Dict] = []
    for stratum in [
        "in-domain",
        "partial-overlap",
        "out-of-domain",
        "unseen_dataset_overlapping_domain",
        "recency",
    ]:
        sub = df[df["sim_stratum"] == stratum]
        if sub.empty:
            continue
        mean_val = float(sub["delta_mase"].mean())
        med_val = float(sub["delta_mase"].median())
        mean_y = float(sub["y_log_mase_ratio"].mean())
        pct_gain_mean = float((1.0 - np.exp(mean_y)) * 100.0)
        win_rate = float((sub["y_log_mase_ratio"] < 0.0).mean())
        cluster_means = sub.groupby("base_dataset")["delta_mase"].mean()
        n_cl = len(cluster_means)
        se_cl = (
            float(cluster_means.std(ddof=1) / np.sqrt(n_cl))
            if n_cl > 1
            else float(sub["delta_mase"].std(ddof=1) / np.sqrt(max(1, len(sub))))
        )
        t_stat = mean_val / max(se_cl, 1e-9)
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, n_cl - 1))))
        strata_rows.append(
            {
                "stratum": stratum,
                "n_pairs": int(len(sub)),
                "n_base_datasets": int(n_cl),
                "inference_status": "inferential (>=5 base datasets)" if n_cl >= 5 else "descriptive_only (<5 base datasets)",
                "mean_log_ratio_y": mean_y,
                "pct_gain_from_y": pct_gain_mean,
                "mean_delta_mase": mean_val,
                "median_delta_mase": med_val,
                "cluster_se": se_cl,
                "ci95_low": mean_val - 1.96 * se_cl,
                "ci95_high": mean_val + 1.96 * se_cl,
                "tsfm_win_rate": win_rate,
                "t_stat_vs_zero": t_stat,
                "p_value_vs_zero": p_val,
            }
        )

    in_dom = df[df["sim_stratum"] == "in-domain"]["delta_mase"].mean()
    for r in strata_rows:
        r["contrast_vs_in_domain"] = float(r["mean_delta_mase"] - in_dom)

    return pd.DataFrame(strata_rows)


def run_phase6_mixed_models(n_boot: int = 1000) -> Dict[str, pd.DataFrame]:
    """Fit M0, M1, M2a, M2b, M2, M3 (ML + REML), cluster bootstrap CIs, GLM checks, and H2 contrasts."""
    out_dir = ensure_dir(PROJECT_ROOT / "results" / "analysis")
    df = build_phase6_analysis_dataset()

    ml_fits = {}
    comparison_rows: List[Dict] = []

    for m_name, formula in MODEL_FORMULAS.items():
        res_ml = _fit_crossed_mixedlm(formula, df, reml=False)
        ml_fits[m_name] = res_ml
        r2_m, r2_c, vc_info = compute_nakagawa_r2(res_ml, df)
        k_params = int(len(res_ml.fe_params) + len(res_ml.vcomp) + 1)
        llf = float(res_ml.llf)
        n_obs = int(len(df))
        aic = float(-2.0 * llf + 2.0 * k_params)
        bic = float(-2.0 * llf + np.log(n_obs) * k_params)

        comparison_rows.append(
            {
                "model_id": m_name,
                "formula": formula,
                "n_obs": n_obs,
                "k_params": k_params,
                "logLik_ML": llf,
                "AIC_ML": aic,
                "BIC_ML": bic,
                "R2_marginal": r2_m,
                "R2_conditional": r2_c,
                **vc_info,
            }
        )

    comp_df = pd.DataFrame(comparison_rows)

    # Likelihood ratio tests (LRT) against nested predecessors
    lrt_pairs = [
        ("M1", "M0"),
        ("M2a", "M1"),
        ("M2b", "M1"),
        ("M2", "M1"),
        ("M3", "M2"),
    ]
    lrt_map: Dict[str, Dict[str, float]] = {}
    for child, parent in lrt_pairs:
        row_c = comp_df[comp_df["model_id"] == child].iloc[0]
        row_p = comp_df[comp_df["model_id"] == parent].iloc[0]
        lr_stat = max(0.0, 2.0 * float(row_c["logLik_ML"] - row_p["logLik_ML"]))
        df_diff = max(1, int(row_c["k_params"] - row_p["k_params"]))
        p_val = float(stats.chi2.sf(lr_stat, df=df_diff))
        lrt_map[child] = {
            "lrt_vs": parent,
            "lrt_chi2": lr_stat,
            "lrt_df": df_diff,
            "lrt_p_value": p_val,
        }

    comp_df["lrt_vs"] = comp_df["model_id"].map(lambda m: lrt_map.get(m, {}).get("lrt_vs", ""))
    comp_df["lrt_chi2"] = comp_df["model_id"].map(lambda m: lrt_map.get(m, {}).get("lrt_chi2", np.nan))
    comp_df["lrt_df"] = comp_df["model_id"].map(lambda m: lrt_map.get(m, {}).get("lrt_df", np.nan))
    comp_df["lrt_p_value"] = comp_df["model_id"].map(lambda m: lrt_map.get(m, {}).get("lrt_p_value", np.nan))
    m2a_aic = float(comp_df.loc[comp_df["model_id"] == "M2a", "AIC_ML"].iloc[0])
    m2a_bic = float(comp_df.loc[comp_df["model_id"] == "M2a", "BIC_ML"].iloc[0])
    comp_df["delta_AIC_vs_M2a"] = comp_df["AIC_ML"] - m2a_aic
    comp_df["delta_BIC_vs_M2a"] = comp_df["BIC_ML"] - m2a_bic
    comp_df.to_csv(out_dir / "mixed_model_comparison.csv", index=False)

    # Compute VIFs including similarity & control predictors (Requirement B.7)
    vif_pred_cols = CHAR_COLS + ["log_params", "dataset_leak", "sim_binary", "sim_share", "sim_distance"]
    X_vif = df[vif_pred_cols].to_numpy(dtype=float)
    X_vif_std = (X_vif - np.mean(X_vif, axis=0)) / np.maximum(np.std(X_vif, axis=0, ddof=1), 1e-12)
    vifs_all = np.diag(np.linalg.pinv(np.corrcoef(X_vif_std, rowvar=False) + 1e-8 * np.eye(len(vif_pred_cols))))
    pd.DataFrame(
        {
            "predictor": vif_pred_cols,
            "vif": [float(max(1.0, v)) for v in vifs_all],
            "keep_under_vif_10": [bool(v < 10.0) for v in vifs_all],
        }
    ).to_csv(PROJECT_ROOT / "results" / "features" / "vif_full_predictors.csv", index=False)

    # Refit M1, M2, M3 with REML for final parameter estimates + cluster bootstrap CIs (B = 1000)
    coef_rows: List[Dict] = []
    for m_name in ["M1", "M2a", "M2b", "M2", "M3"]:
        formula = MODEL_FORMULAS[m_name]
        res_reml = _fit_crossed_mixedlm(formula, df, reml=True)
        boot_cis = cluster_bootstrap_fixed_effects(
            formula, df, n_boot=n_boot if m_name in ("M2", "M3") else 250, res_reml=res_reml
        )

        # Also fit cluster-robust Gaussian GLM as a diagnostic check
        glm_res = smf.glm(formula, data=df, family=sm.families.Gaussian()).fit(
            cov_type="cluster", cov_kwds={"groups": df["base_dataset"]}
        )

        fe_names = list(res_reml.fe_params.index)
        for term in fe_names:
            est = float(res_reml.fe_params[term])
            se = float(res_reml.bse_fe.get(term, res_reml.bse.get(term, np.nan)))
            t_val = est / max(se, 1e-12)
            p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_val))))
            b_low, b_high = boot_cis.get(term, (est - 1.96 * se, est + 1.96 * se))
            glm_se = float(glm_res.bse.get(term, np.nan))
            glm_p = float(glm_res.pvalues.get(term, np.nan))

            coef_rows.append(
                {
                    "model_id": m_name,
                    "term": term,
                    "estimate_reml": est,
                    "estimate_log_ratio_y": -est,
                    "pct_gain_equiv": float((1.0 - np.exp(-est)) * 100.0),
                    "se_reml": se,
                    "z_stat": t_val,
                    "p_value": p_val,
                    "boot_ci95_low": b_low,
                    "boot_ci95_high": b_high,
                    "glm_cluster_se": glm_se,
                    "glm_cluster_p_value": glm_p,
                }
            )

    coef_df = pd.DataFrame(coef_rows)

    # Apply Benjamini-Hochberg FDR across domain-similarity block coefficients in M2 and M3
    coef_df["fdr_bh_q_value"] = np.nan
    for m_name in ["M2", "M3"]:
        mask = (coef_df["model_id"] == m_name) & (
            coef_df["term"].str.contains("sim_", regex=False)
        )
        if mask.any():
            pvals = coef_df.loc[mask, "p_value"].to_numpy(dtype=float)
            _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            coef_df.loc[mask, "fdr_bh_q_value"] = qvals

    coef_df.to_csv(out_dir / "mixed_model_coefficients.csv", index=False)

    # Robustness checks: log_mase_ratio, delta_crps, and Gamma GLM on mase / best_classical_mase
    rob_rows: List[Dict] = []
    for resp_name, resp_formula in [
        ("log_mase_ratio", MODEL_FORMULAS["M2"].replace("delta_mase ~", "log_mase_ratio ~")),
        ("delta_crps", MODEL_FORMULAS["M2"].replace("delta_mase ~", "delta_crps ~")),
        ("knn_similarity", MODEL_FORMULAS["M2"].replace("z_sim_distance", "z_sim_distance_knn")),
    ]:
        res_rob = _fit_crossed_mixedlm(resp_formula, df, reml=True)
        for term in ["sim_binary", "z_sim_share", "z_sim_distance", "z_sim_distance_knn", "z_forecastability", "z_seasonal_strength"]:
            if term in res_rob.fe_params:
                est = float(res_rob.fe_params[term])
                se = float(res_rob.bse_fe.get(term, res_rob.bse.get(term, np.nan)))
                rob_rows.append(
                    {
                        "specification": resp_name,
                        "term": term,
                        "estimate": est,
                        "se": se,
                        "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(est / max(se, 1e-12))))),
                    }
                )

    # Gamma GLM diagnostic on relative MASE ratio (mase / best_classical_mase)
    df["mase_ratio_pos"] = np.clip(df["mase"] / np.maximum(df["best_classical_mase"], 1e-6), 0.05, 10.0)
    gamma_formula = MODEL_FORMULAS["M2"].replace("delta_mase ~", "mase_ratio_pos ~")
    gamma_res = smf.glm(
        gamma_formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit(cov_type="cluster", cov_kwds={"groups": df["base_dataset"]})
    for term in ["sim_binary", "z_sim_share", "z_sim_distance", "z_forecastability", "z_seasonal_strength"]:
        if term in gamma_res.params:
            rob_rows.append(
                {
                    "specification": "gamma_glm_log_link",
                    "term": term,
                    "estimate": float(gamma_res.params[term]),
                    "se": float(gamma_res.bse[term]),
                    "p_value": float(gamma_res.pvalues[term]),
                }
            )

    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(out_dir / "mixed_model_robustness.csv", index=False)

    # H2 strata contrasts
    h2_df = evaluate_h2_strata_contrasts(df)
    h2_df.to_csv(out_dir / "h2_strata_contrasts.csv", index=False)

    return {
        "comparison": comp_df,
        "coefficients": coef_df,
        "robustness": rob_df,
        "h2_contrasts": h2_df,
    }


def recover_planted_synthetic_effect(
    true_beta_sim_distance: float = -0.15,
    seed: int = 20260410,
) -> Dict[str, float]:
    """Generate a synthetic (model, config) dataset with known beta_sim_distance and test recovery."""
    rng = np.random.default_rng(seed)
    n_datasets = 25
    n_configs_per_ds = 4
    n_models = 15

    ds_effects = rng.normal(0.0, 0.08, size=n_datasets)
    fam_effects = rng.normal(0.0, 0.05, size=5)

    rows = []
    for d_idx in range(n_datasets):
        base_ds = f"syn_ds_{d_idx:02d}"
        for c_idx in range(n_configs_per_ds):
            cfg = f"{base_ds}/D/short_{c_idx}"
            z_fc = float(rng.normal(0.0, 1.0))
            z_seas = float(rng.normal(0.0, 1.0))
            z_trend = float(rng.normal(0.0, 1.0))
            z_len = float(rng.normal(0.0, 1.0))
            for m_idx in range(n_models):
                fam_idx = m_idx % 5
                sim_bin = int(rng.integers(0, 2))
                z_sim_dist = float(rng.normal(0.0, 1.0))
                eps = float(rng.normal(0.0, 0.08))
                y = (
                    0.05
                    + 0.06 * sim_bin
                    + true_beta_sim_distance * z_sim_dist
                    + 0.04 * z_fc
                    + 0.03 * z_seas
                    - 0.02 * z_trend
                    + 0.01 * z_len
                    + ds_effects[d_idx]
                    + fam_effects[fam_idx]
                    + eps
                )
                rows.append(
                    {
                        "base_dataset": base_ds,
                        "config": cfg,
                        "model": f"model_{m_idx}",
                        "model_family": f"fam_{fam_idx}",
                        "sim_binary": sim_bin,
                        "z_sim_distance": z_sim_dist,
                        "z_forecastability": z_fc,
                        "z_seasonal_strength": z_seas,
                        "z_trend_strength": z_trend,
                        "z_log_length": z_len,
                        "delta_mase": y,
                    }
                )
    syn_df = pd.DataFrame(rows)
    formula = (
        "delta_mase ~ sim_binary + z_sim_distance + z_forecastability "
        "+ z_seasonal_strength + z_trend_strength + z_log_length"
    )
    res = _fit_crossed_mixedlm(formula, syn_df, reml=True)
    est = float(res.fe_params["z_sim_distance"])
    se = float(res.bse_fe.get("z_sim_distance", res.bse["z_sim_distance"]))
    ci_low = est - 1.96 * se
    ci_high = est + 1.96 * se
    return {
        "true_beta": true_beta_sim_distance,
        "estimated_beta": est,
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "covered": float(ci_low <= true_beta_sim_distance <= ci_high),
    }
