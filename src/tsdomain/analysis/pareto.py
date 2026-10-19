"""Phase 9: Compute-Efficiency Pareto Frontier Analysis & H4 Decision Boundary."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from tsdomain.utils import PROJECT_ROOT, ensure_dir


def _identify_pareto_frontier(
    mase_vals: np.ndarray,
    cost_vals: np.ndarray,
) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points minimizing both MASE and compute cost."""
    n = len(mase_vals)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(mase_vals[i]) or not np.isfinite(cost_vals[i]):
            is_pareto[i] = False
            continue
        for j in range(n):
            if i == j or not np.isfinite(mase_vals[j]) or not np.isfinite(cost_vals[j]):
                continue
            # j dominates i if j is <= i on both and strictly < on at least one
            if (
                mase_vals[j] <= mase_vals[i]
                and cost_vals[j] <= cost_vals[i]
                and (mase_vals[j] < mase_vals[i] or cost_vals[j] < cost_vals[i])
            ):
                is_pareto[i] = False
                break
    return is_pareto


def run_pareto_analysis() -> Dict[str, pd.DataFrame]:
    """Compute Pareto frontier, marginal energy per 1% MASE gain, and H4 decision boundary."""
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    compute_dir = PROJECT_ROOT / "results" / "compute"
    features_dir = PROJECT_ROOT / "results" / "features"
    analysis_dir = ensure_dir(PROJECT_ROOT / "results" / "analysis")

    cls_df = pd.read_parquet(metrics_dir / "classical_baselines.parquet")
    best_cls_df = pd.read_parquet(metrics_dir / "best_classical_local.parquet")
    tsfm_df = pd.read_parquet(metrics_dir / "tsfm_local_metrics.parquet")
    comp_df = pd.read_parquet(compute_dir / "compute_profile.parquet")

    # Combine classical and TSFM accuracy on the 13 local configs
    cls_sub = cls_df[["model", "config", "domain", "freq", "mase", "crps"]].copy()
    cls_sub["model_family_group"] = "classical"

    tsfm_sub = tsfm_df[["model", "config", "domain", "freq", "mase", "crps"]].copy()
    tsfm_sub["model_family_group"] = "tsfm"

    # Include LightGBM if Phase 8 has already been run
    lgbm_path = metrics_dir / "lgbm_local_metrics.parquet"
    if lgbm_path.exists():
        lgbm_df = pd.read_parquet(lgbm_path)[["model", "config", "domain", "freq", "mase", "crps"]].copy()
        lgbm_df["model_family_group"] = "ml_tabular"
        acc_df = pd.concat([cls_sub, tsfm_sub, lgbm_df], ignore_index=True)
    else:
        acc_df = pd.concat([cls_sub, tsfm_sub], ignore_index=True)

    merged = acc_df.merge(
        comp_df[
            [
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
        ],
        on=["model", "config"],
        how="inner",
    )

    # Join best classical baseline MASE and energy per config
    best_cls_energy = (
        merged[merged["model"].isin(["SeasonalNaive", "AutoETS", "AutoTheta", "AutoARIMA"])]
        .sort_values("mase")
        .groupby("config", as_index=False)
        .first()[
            [
                "config",
                "model",
                "mase",
                "infer_time_ms_per_point",
                "energy_mwh_per_1k_points",
            ]
        ]
        .rename(
            columns={
                "model": "best_classical_model",
                "mase": "best_classical_mase",
                "infer_time_ms_per_point": "best_classical_ms_per_pt",
                "energy_mwh_per_1k_points": "best_classical_energy_mwh",
            }
        )
    )

    merged = merged.merge(best_cls_energy, on="config", how="left")
    merged["inv_mase"] = 1.0 / np.maximum(merged["mase"], 1e-6)
    merged["delta_mase"] = (
        merged["best_classical_mase"] - merged["mase"]
    ) / merged["best_classical_mase"]
    merged["delta_mase_pct"] = 100.0 * merged["delta_mase"]

    # Marginal energy cost per 1% MASE gain (mWh per 1k points per 1% MASE improvement)
    delta_energy = merged["energy_mwh_per_1k_points"] - merged["best_classical_energy_mwh"]
    gain_pct = merged["delta_mase_pct"]
    merged["dominated_by_classical"] = gain_pct <= 0.0
    merged["energy_per_1pct_mase_gain_mwh"] = np.where(
        gain_pct > 0.0,
        np.maximum(0.0, delta_energy) / np.maximum(gain_pct, 1e-6),
        np.nan,
    )
    merged["speedup_vs_best_classical"] = (
        merged["best_classical_ms_per_pt"] / np.maximum(merged["infer_time_ms_per_point"], 1e-6)
    )

    # Identify per-config Pareto frontier on (MASE, energy_mwh_per_1k_points) and (MASE, infer_time_ms_per_point)
    merged["pareto_energy"] = False
    merged["pareto_latency"] = False
    for cfg, grp in merged.groupby("config"):
        pe = _identify_pareto_frontier(
            grp["mase"].to_numpy(dtype=float),
            grp["energy_mwh_per_1k_points"].to_numpy(dtype=float),
        )
        pl = _identify_pareto_frontier(
            grp["mase"].to_numpy(dtype=float),
            grp["infer_time_ms_per_point"].to_numpy(dtype=float),
        )
        merged.loc[grp.index, "pareto_energy"] = pe
        merged.loc[grp.index, "pareto_latency"] = pl

    merged.to_csv(analysis_dir / "pareto_table.csv", index=False)

    # Model-level Pareto summary averaged across all 13 local configs
    model_summary = (
        merged.groupby(["model", "model_family_group"], as_index=False)
        .agg(
            param_count=("param_count", "first"),
            mean_mase=("mase", "mean"),
            median_mase=("mase", "median"),
            mean_crps=("crps", "mean"),
            mean_delta_mase_pct=("delta_mase_pct", "mean"),
            win_rate_vs_best_classical=("delta_mase", lambda s: float((s > 0).mean())),
            mean_infer_ms_per_point=("infer_time_ms_per_point", "mean"),
            median_infer_ms_per_point=("infer_time_ms_per_point", "median"),
            mean_peak_ram_mb=("peak_ram_mb", "mean"),
            mean_energy_mwh_per_1k_pts=("energy_mwh_per_1k_points", "mean"),
            median_energy_per_1pct_gain_mwh=("energy_per_1pct_mase_gain_mwh", "median"),
            pareto_config_share_energy=("pareto_energy", "mean"),
        )
        .sort_values("mean_mase")
        .reset_index(drop=True)
    )
    model_summary["global_pareto_energy"] = _identify_pareto_frontier(
        model_summary["mean_mase"].to_numpy(dtype=float),
        model_summary["mean_energy_mwh_per_1k_pts"].to_numpy(dtype=float),
    )
    model_summary["global_pareto_latency"] = _identify_pareto_frontier(
        model_summary["mean_mase"].to_numpy(dtype=float),
        model_summary["mean_infer_ms_per_point"].to_numpy(dtype=float),
    )
    model_summary.to_csv(analysis_dir / "pareto_summary_by_model.csv", index=False)

    # H4 test: Map feature-space decision boundary from fitted M2/M3 predictions across all 99 configs
    p6_path = analysis_dir / "phase6_analysis_dataset.parquet"
    lodo_path = analysis_dir / "lodo_cv_predictions.parquet"
    if p6_path.exists() and lodo_path.exists():
        p6_df = pd.read_parquet(p6_path)
        lodo_df = pd.read_parquet(lodo_path)
        h4_df = p6_df.merge(
            lodo_df[["model", "config", "pred_M1", "pred_M2", "pred_M3"]],
            on=["model", "config"],
            how="left",
        )
        # Define 3x3 grid over (sim_distance tertiles x forecastability tertiles) and length regimes
        h4_df["sim_dist_regime"] = pd.qcut(
            h4_df["sim_distance"], q=3, labels=["Low_Shift_InDomain", "Moderate_Shift", "High_Shift_OOD"]
        )
        h4_df["forecastability_regime"] = pd.qcut(
            h4_df["forecastability"], q=3, labels=["Low_Forecastability", "Med_Forecastability", "High_Forecastability"]
        )
        h4_df["length_regime"] = np.where(h4_df["length"] < 700, "Short_lt_700", "Long_ge_700")

        grid_summary = (
            h4_df.groupby(["sim_dist_regime", "forecastability_regime", "length_regime"], observed=True, as_index=False)
            .agg(
                n_pairs=("delta_mase", "count"),
                actual_mean_delta_mase_pct=("delta_mase", lambda s: float(100.0 * s.mean())),
                predicted_M2_delta_mase_pct=("pred_M2", lambda s: float(100.0 * s.mean())),
                predicted_M3_delta_mase_pct=("pred_M3", lambda s: float(100.0 * s.mean())),
                tsfm_win_rate=("tsfm_wins", "mean"),
            )
        )
        grid_summary["classical_pareto_optimal_0pct"] = grid_summary["predicted_M3_delta_mase_pct"] <= 0.0
        grid_summary["classical_pareto_optimal_2pct"] = grid_summary["predicted_M3_delta_mase_pct"] < 2.0
        grid_summary.to_csv(analysis_dir / "h4_decision_boundary.csv", index=False)
    else:
        grid_summary = pd.DataFrame()

    return {
        "pareto_table": merged,
        "model_summary": model_summary,
        "h4_boundary": grid_summary,
    }
