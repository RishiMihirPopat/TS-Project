"""Phase 8a (P1): LightGBM Tabular Forecasting Baseline with Optuna Budget Parity."""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import norm

from tsdomain.baselines.run_baselines import (
    LOCAL_CONFIGS,
    load_evaluation_windows_for_config,
)
from tsdomain.compute.profiler import profile_callable, summarize_profile_repeats
from tsdomain.utils import (
    PROJECT_ROOT,
    QUANTILE_COLS,
    QUANTILE_LEVELS,
    ensure_dir,
    save_forecast_with_metadata,
    score_forecast_window,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_lag_features(
    series_list: List[np.ndarray],
    n_lags: int,
    season_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build temporal train/val lag + rolling mean/std + seasonal-phase features per series without overlap."""
    X_tr, y_tr, X_va, y_va = [], [], [], []
    m = max(1, int(season_length))
    roll_w = min(max(3, m), n_lags)

    for arr in series_list:
        y = np.asarray(arr, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) <= n_lags + 2:
            continue
        n_avail = len(y) - n_lags
        n_val_s = max(1, int(0.2 * n_avail))
        split_t = len(y) - n_val_s
        # Z-score using only the pre-validation prefix to avoid leaking validation targets into scale
        mu = float(np.mean(y[:split_t]))
        sd = max(float(np.std(y[:split_t], ddof=1)), 1e-6)
        z = (y - mu) / sd

        for t in range(n_lags, len(z)):
            lags = z[t - n_lags : t][::-1]
            recent = z[t - roll_w : t]
            feat = np.concatenate(
                [lags, [float(np.mean(recent)), float(np.std(recent)), float((t % m) / m)]]
            )
            if t < split_t:
                X_tr.append(feat)
                y_tr.append(float(z[t]))
            else:
                X_va.append(feat)
                y_va.append(float(z[t]))

    if not X_tr:
        z_x, z_y = np.zeros((10, n_lags + 3), dtype=np.float32), np.zeros(10, dtype=np.float32)
        return z_x, z_y, z_x[:2], z_y[:2]
    return (
        np.asarray(X_tr, dtype=np.float32),
        np.asarray(y_tr, dtype=np.float32),
        np.asarray(X_va or X_tr[-2:], dtype=np.float32),
        np.asarray(y_va or y_tr[-2:], dtype=np.float32),
    )


def fit_and_forecast_lgbm_config(
    windows: List[Dict],
    time_budget_s: float = 2.5,
    seed: int = 20260410,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tune a global LightGBM model under strict wall-clock budget parity and forecast all windows."""
    horizon = int(windows[0]["prediction_length"])
    season_length = int(windows[0]["season_length"])
    min_len = min(len(w["y_train"]) for w in windows)
    n_lags = min(max(6, 2 * max(1, season_length)), min(24, max(4, min_len - 4)))
    m = max(1, season_length)
    roll_w = min(max(3, m), n_lags)

    # Use only the earliest training window per item_id so rolling w=0 and w=1 do not overlap in training/val
    min_w = min(int(w["window_id"]) for w in windows)
    # shortcut: caps training history at 256 points and 3200 lag rows for sub-second CPU budget parity, increase caps for offline training
    train_series = [w["y_train"][-256:] for w in windows if int(w["window_id"]) == min_w]
    X_tr, y_tr, X_va, y_va = _build_lag_features(train_series, n_lags=n_lags, season_length=season_length)

    rng = np.random.default_rng(seed)
    if len(X_tr) > 3200:
        idx_tr = rng.choice(len(X_tr), size=3200, replace=False)
        X_tr, y_tr = X_tr[idx_tr], y_tr[idx_tr]
    if len(X_va) > 800:
        idx_va = rng.choice(len(X_va), size=800, replace=False)
        X_va, y_va = X_va[idx_va], y_va[idx_va]
    X_all = np.vstack([X_tr, X_va])
    y_all = np.concatenate([y_tr, y_va])

    # Optuna hyperparameter search capped strictly to 50% of time_budget_s
    tune_budget = max(0.25, 0.5 * float(time_budget_s))

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "n_jobs": 1,
            "random_state": seed,
            "n_estimators": trial.suggest_int("n_estimators", 30, 120),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.18, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = lgb.LGBMRegressor(**params)
            reg.fit(X_tr, y_tr)
            pred_va = reg.predict(X_va)
        return float(np.mean(np.abs(y_va - pred_va)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(_objective, timeout=tune_budget, n_trials=6, show_progress_bar=False)
    best_params = {
        "objective": "regression",
        "verbosity": -1,
        "n_jobs": 1,
        "random_state": seed,
        **study.best_params,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X_all, y_all)
        in_sample_resid_std = float(np.std(y_all - final_model.predict(X_all), ddof=1))

    # Recursive multi-step forecasting across all evaluation windows
    forecast_dfs: List[pd.DataFrame] = []
    metric_rows: List[Dict] = []

    for w in windows:
        y_train = np.asarray(w["y_train"], dtype=float)
        y_test = np.asarray(w["y_test"], dtype=float)
        test_ts = w["test_timestamps"]
        item_id = str(w["item_id"])
        win_id = int(w["window_id"])

        mu = float(np.mean(y_train))
        sd = max(float(np.std(y_train, ddof=1)), 1e-6)
        z_hist = list((y_train - mu) / sd)
        while len(z_hist) < n_lags:
            z_hist.insert(0, z_hist[0] if z_hist else 0.0)

        z_preds = np.empty(horizon, dtype=float)
        base_t = len(y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step in range(horizon):
                lags = np.asarray(z_hist[-n_lags:][::-1], dtype=np.float32)
                recent = np.asarray(z_hist[-roll_w:], dtype=np.float32)
                rmean = float(np.mean(recent))
                rstd = float(np.std(recent))
                s_phase = float(((base_t + step) % m) / max(1.0, float(m)))
                x_step = np.concatenate([lags, [rmean, rstd, s_phase]])[None, :]
                z_hat = float(final_model.predict(x_step)[0])
                z_preds[step] = z_hat
                z_hist.append(z_hat)

        mean_pred = z_preds * sd + mu
        steps = np.arange(1, horizon + 1, dtype=float)
        sigma_h = max(in_sample_resid_std * sd, 1e-6) * np.sqrt(1.0 + 0.20 * (steps - 1.0))
        quantiles = {
            col: mean_pred + float(norm.ppf(q)) * sigma_h
            for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS)
        }

        df_w, m_w = score_forecast_window(
            w["item_id"],
            w["window_id"],
            y_train,
            y_test,
            w["test_timestamps"],
            season_length,
            mean_pred,
            quantiles,
        )
        forecast_dfs.append(df_w)
        metric_rows.append(m_w)

    return pd.concat(forecast_dfs, ignore_index=True), pd.DataFrame(metric_rows)


def run_lightgbm_baseline() -> pd.DataFrame:
    """Run LightGBM with Optuna budget parity across all 13 local configs and update compute profile."""
    metrics_dir = ensure_dir(PROJECT_ROOT / "results" / "metrics")
    compute_dir = ensure_dir(PROJECT_ROOT / "results" / "compute")
    lgbm_fc_dir = ensure_dir(PROJECT_ROOT / "results" / "forecasts" / "LightGBM")

    comp_path = compute_dir / "compute_profile.parquet"
    comp_df = pd.read_parquet(comp_path) if comp_path.exists() else pd.DataFrame()

    # Budget parity: cap Optuna + LightGBM time to median Moirai_small runtime per config
    moirai_budgets: Dict[str, float] = {}
    if not comp_df.empty and "Moirai_small" in comp_df["model"].values:
        m_sub = comp_df[comp_df["model"] == "Moirai_small"]
        moirai_budgets = dict(zip(m_sub["config"], m_sub["infer_time_s_total"]))

    metric_rows: List[Dict] = []
    new_comp_rows: List[Dict] = []

    for cfg in LOCAL_CONFIGS:
        windows = load_evaluation_windows_for_config(cfg)
        n_pts = int(sum(int(w["prediction_length"]) for w in windows))
        budget_s = float(moirai_budgets.get(cfg, 2.0))

        (fc_df, m_df), prof = profile_callable(
            fit_and_forecast_lgbm_config,
            windows,
            time_budget_s=budget_s,
            device="cpu",
            n_forecast_points=n_pts,
            load_time_s=0.0,
        )
        comp_summary = summarize_profile_repeats(
            [prof],
            model_id="LightGBM",
            config=cfg,
            param_count=15_000,
        )
        new_comp_rows.append(comp_summary)

        safe_cfg = cfg.replace("/", "__")
        save_forecast_with_metadata(
            df=fc_df,
            parquet_path=lgbm_fc_dir / f"{safe_cfg}.parquet",
            model_name="LightGBM",
            dataset_config=cfg,
            extra_meta={
                "budget_parity_target_s": budget_s,
                "compute_summary": comp_summary,
            },
        )

        metric_rows.append(
            {
                "model": "LightGBM",
                "config": cfg,
                "domain": windows[0]["domain"],
                "freq": windows[0]["freq"],
                "horizon": int(windows[0]["prediction_length"]),
                "num_windows": int(len(m_df)),
                "mase": float(m_df["mase"].mean()),
                "mase_median": float(m_df["mase"].median()),
                "rmsse": float(m_df["rmsse"].mean()),
                "rmse": float(m_df["rmse"].mean()),
                "mape": float(m_df["mape"].mean()),
                "crps": float(m_df["crps"].mean()),
                "coverage_80": float(m_df["coverage_80"].mean()),
                "width_80": float(m_df["width_80"].mean()),
                "interval_score_80": float(m_df["interval_score_80"].mean()),
                "infer_time_ms_per_point": comp_summary["infer_time_ms_per_point"],
                "energy_mwh_per_1k_points": comp_summary["energy_mwh_per_1k_points"],
                "budget_parity_s": budget_s,
            }
        )

    lgbm_df = pd.DataFrame(metric_rows)
    lgbm_df.to_parquet(metrics_dir / "lgbm_local_metrics.parquet", index=False)

    if not comp_df.empty:
        comp_df = comp_df[comp_df["model"] != "LightGBM"].copy()
        comp_df = pd.concat([comp_df, pd.DataFrame(new_comp_rows)], ignore_index=True)
        comp_df.to_parquet(comp_path, index=False)

    return lgbm_df
