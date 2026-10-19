"""Phase 2: Run classical baselines across unseen datasets and GIFT-Eval subset, and verify leaderboard parity."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tsdomain.baselines.classical import run_classical_on_dataset
from tsdomain.config import load_and_validate_configs
from tsdomain.data.gift_eval import generate_rolling_windows
from tsdomain.metrics.point import aggregate_config_metrics, mase

CORE_FOUR_CLASSICAL = ["SeasonalNaive", "AutoETS", "AutoTheta", "AutoARIMA"]
ALL_CLASSICAL_MODELS = ["Naive", "SeasonalNaive", "Drift", "AutoETS", "AutoTheta", "AutoARIMA"]
LEADERBOARD_FOUR_CLASSICAL = ["Seasonal_Naive", "Auto_ETS", "Auto_Theta", "Auto_Arima"]

LOCAL_CONFIGS = [
    "m4_monthly/M/short",
    "m4_quarterly/Q/short",
    "ett1/D/short",
    "solar/W/short",
    "hospital/M/short",
    "us_births/M/short",
    "saugeen/W/short",
    "jena_weather/D/short",
    "hierarchical_sales/W/short",
    "loop_seattle/D/short",
    "bizitobs_l2c/H/short",
    "unseen_mandi/W/short",
    "unseen_recency/D/short",
]


def load_evaluation_windows_for_config(config_name: str) -> List[Dict]:
    """Load rolling evaluation windows for a local config with domain and base_dataset metadata."""
    cfgs = load_and_validate_configs()
    exp_cfg = cfgs["experiment"]
    ds_map = {d.config: d for d in cfgs["datasets"].datasets}
    ds_entry = ds_map.get(config_name)
    windows, _ = generate_rolling_windows(
        config_name,
        max_series=exp_cfg.max_series_per_config,
        max_windows_per_series=exp_cfg.max_windows_per_series,
        seed=exp_cfg.seed,
        exact_gift_eval=False,
    )
    for w in windows:
        w["domain"] = ds_entry.domain if ds_entry else "unknown"
        w["base_dataset"] = ds_entry.base_dataset if ds_entry else config_name.split("/")[0]
    return windows


def run_leaderboard_crosscheck(
    configs: List[str],
    leaderboard_parquet: Path = Path("results/metrics/leaderboard.parquet"),
    out_parquet: Path = Path("results/metrics/leaderboard_crosscheck.parquet"),
) -> pd.DataFrame:
    """Reproduce GIFT-Eval leaderboard Seasonal_Naive MASE on full configurations."""
    lb = pd.read_parquet(leaderboard_parquet)
    sn_lb = lb[lb["model"] == "Seasonal_Naive"].set_index("config")["mase"].to_dict()

    rows: List[Dict] = []
    for cfg in configs:
        wins, info = generate_rolling_windows(cfg, exact_gift_eval=True)
        m = info["season_length"]
        mases = []
        for w in wins:
            y_tr, y_te = w["y_train"], w["y_test"]
            h = len(y_te)
            cycle = y_tr[-m:]
            y_pred = np.tile(cycle, int(np.ceil(h / m)))[:h]
            mases.append(mase(y_te, y_pred, y_tr, m=m))
        local_mase = float(np.mean(mases))
        lb_mase = float(sn_lb[cfg])
        abs_diff = abs(local_mase - lb_mase)
        rows.append(
            {
                "config": cfg,
                "n_series": info["n_series"],
                "n_eval_windows": info["total_eval_windows"],
                "season_length": m,
                "local_seasonal_naive_mase": local_mase,
                "leaderboard_seasonal_naive_mase": lb_mase,
                "abs_diff": abs_diff,
                "rel_diff_pct": (abs_diff / max(lb_mase, 1e-12)) * 100.0,
                "passed": bool(abs_diff < 1e-4),
            }
        )

    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_parquet.with_suffix(".csv"), index=False)
    return df


def compute_best_classical_from_leaderboard(
    leaderboard_parquet: Path = Path("results/metrics/leaderboard.parquet"),
    out_parquet: Path = Path("results/metrics/best_classical_leaderboard.parquet"),
) -> pd.DataFrame:
    """Compute best_classical per config across all 97 GIFT-Eval configs from the 4 classical models."""
    lb = pd.read_parquet(leaderboard_parquet)
    sub = lb[lb["model"].isin(LEADERBOARD_FOUR_CLASSICAL)].copy()

    rows = []
    for cfg, grp in sub.groupby("config", sort=True):
        grp_valid = grp.dropna(subset=["mase"]).sort_values("mase")
        best_row = grp_valid.iloc[0]
        rows.append(
            {
                "config": str(cfg),
                "base_dataset": str(best_row["base_dataset"]),
                "domain": str(best_row["domain"]),
                "freq": str(best_row["freq"]),
                "term": str(best_row["term"]),
                "best_classical_model": str(best_row["model"]),
                "best_classical_mase": float(best_row["mase"]),
                "best_classical_crps": float( grp_valid["crps"].min() ),
                "seasonal_naive_mase": float(
                    grp[grp["model"] == "Seasonal_Naive"]["mase"].iloc[0]
                ),
                "auto_ets_mase": float(grp[grp["model"] == "Auto_ETS"]["mase"].iloc[0]),
                "auto_theta_mase": float(grp[grp["model"] == "Auto_Theta"]["mase"].iloc[0]),
                "auto_arima_mase": float(grp[grp["model"] == "Auto_Arima"]["mase"].iloc[0]),
            }
        )
    df_out = pd.DataFrame(rows).sort_values("config").reset_index(drop=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_parquet, index=False)
    return df_out


def run_phase2_baselines() -> None:
    """Execute Phase 2: Leaderboard cross-check and classical baseline evaluation on local configs."""
    cfgs = load_and_validate_configs()
    exp_cfg = cfgs["experiment"]
    ds_map = {d.config: d for d in cfgs["datasets"].datasets}

    print("[Phase 2.1] Running GIFT-Eval Seasonal_Naive leaderboard cross-check...")
    cc_df = run_leaderboard_crosscheck(exp_cfg.leaderboard_crosscheck_configs)
    for _, r in cc_df.iterrows():
        print(
            f"  -> {r['config']:22s}: local={r['local_seasonal_naive_mase']:.6f}, "
            f"leaderboard={r['leaderboard_seasonal_naive_mase']:.6f}, diff={r['abs_diff']:.6f}"
        )

    print("[Phase 2.2] Computing best_classical across all 97 GIFT-Eval leaderboard configs...")
    lb_best_df = compute_best_classical_from_leaderboard()
    print(f"  -> Saved best_classical_leaderboard.parquet ({len(lb_best_df)} configs).")

    print(f"[Phase 2.3] Running 6 classical baselines across {len(exp_cfg.local_eval_configs)} local configs...")
    summary_rows: List[Dict] = []

    for cfg_name in exp_cfg.local_eval_configs:
        ds_entry = ds_map[cfg_name]
        windows, win_info = generate_rolling_windows(
            cfg_name,
            max_series=exp_cfg.max_series_per_config,
            max_windows_per_series=exp_cfg.max_windows_per_series,
            seed=exp_cfg.seed,
            exact_gift_eval=False,
        )
        season_len = win_info["season_length"]

        for model_name in ALL_CLASSICAL_MODELS:
            _, win_metrics_df = run_classical_on_dataset(
                windows=windows,
                dataset_config=cfg_name,
                season_length=season_len,
                model_name=model_name,
                seed=exp_cfg.seed,
            )
            agg = aggregate_config_metrics(win_metrics_df)
            summary_rows.append(
                {
                    "config": cfg_name,
                    "base_dataset": ds_entry.base_dataset,
                    "domain": ds_entry.domain,
                    "freq": ds_entry.freq,
                    "term": ds_entry.term,
                    "model": model_name,
                    "model_type": "statistical",
                    "n_series": win_info["n_series"],
                    "n_windows": win_info["total_eval_windows"],
                    "season_length": season_len,
                    **agg,
                }
            )
        print(f"  -> Completed classical baselines for {cfg_name} ({win_info['total_eval_windows']} windows)")

    classical_df = pd.DataFrame(summary_rows)
    out_metrics = Path("results/metrics/classical_baselines.parquet")
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    classical_df.to_parquet(out_metrics, index=False)
    classical_df.to_csv(out_metrics.with_suffix(".csv"), index=False)

    # Compute best_classical per local config over the 4 core classical models
    core_sub = classical_df[classical_df["model"].isin(CORE_FOUR_CLASSICAL)].copy()
    best_local_rows = []
    for cfg_name, grp in core_sub.groupby("config", sort=True):
        grp_sorted = grp.sort_values("mase")
        best_r = grp_sorted.iloc[0]
        best_local_rows.append(
            {
                "config": str(cfg_name),
                "base_dataset": str(best_r["base_dataset"]),
                "domain": str(best_r["domain"]),
                "freq": str(best_r["freq"]),
                "term": str(best_r["term"]),
                "best_classical_model": str(best_r["model"]),
                "best_classical_mase": float(best_r["mase"]),
                "best_classical_crps": float(grp_sorted["crps"].min()),
                "seasonal_naive_mase": float(
                    grp[grp["model"] == "SeasonalNaive"]["mase"].iloc[0]
                ),
                "auto_ets_mase": float(grp[grp["model"] == "AutoETS"]["mase"].iloc[0]),
                "auto_theta_mase": float(grp[grp["model"] == "AutoTheta"]["mase"].iloc[0]),
                "auto_arima_mase": float(grp[grp["model"] == "AutoARIMA"]["mase"].iloc[0]),
            }
        )
    best_local_df = pd.DataFrame(best_local_rows)
    best_local_path = Path("results/metrics/best_classical_local.parquet")
    best_local_df.to_parquet(best_local_path, index=False)
    best_local_df.to_csv(best_local_path.with_suffix(".csv"), index=False)
    print(f"  -> Saved classical_baselines.parquet and best_classical_local.parquet ({len(best_local_df)} configs).")
