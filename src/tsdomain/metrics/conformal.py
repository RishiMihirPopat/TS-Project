"""Phase 5 (P1): Native Quantile & Time-Ordered Split Conformal Calibration (CQR & Residual Conformal)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from tsdomain.baselines.run_baselines import LOCAL_CONFIGS
from tsdomain.metrics.probabilistic import (
    coverage_and_width,
    crps_weighted_quantile_loss,
    interval_score,
)
from tsdomain.utils import PROJECT_ROOT, QUANTILE_COLS, QUANTILE_LEVELS, ensure_dir

NATIVE_QUANTILE_MODELS = {
    "Chronos_small",
    "Moirai_small",
    "TimesFM",
    "AutoETS",
    "AutoTheta",
    "AutoARIMA",
    "SeasonalNaive",
}

POINT_CONFORMAL_MODELS = {
    "TTM-R2-Pretrained",
    "Naive",
    "Drift",
    "LightGBM",
}


def compute_cqr_offset(
    y_cal: np.ndarray,
    q_low_cal: np.ndarray,
    q_high_cal: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """Compute the Conformalized Quantile Regression (CQR; Romano et al., 2019) calibration offset.

    Nonconformity score: s_i = max(q_low_i - y_i, y_i - q_high_i).
    Returns the finite-sample corrected empirical quantile Q_{(1-alpha)(1 + 1/n_cal)}(s).
    """
    y = np.asarray(y_cal, dtype=float)
    lo = np.asarray(q_low_cal, dtype=float)
    hi = np.asarray(q_high_cal, dtype=float)
    scores = np.maximum(lo - y, y - hi)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / float(n)))
    return float(np.quantile(scores, level, method="higher"))


def compute_abs_residual_conformal_halfwidth(
    y_cal: np.ndarray,
    mean_cal: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """Compute standard split conformal half-width on absolute residuals |y_i - y_hat_i|."""
    y = np.asarray(y_cal, dtype=float)
    mu = np.asarray(mean_cal, dtype=float)
    scores = np.abs(y - mu)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / float(n)))
    return float(np.quantile(scores, level, method="higher"))


def apply_conformal_to_forecast_df(
    fc_df: pd.DataFrame,
    method: str = "cqr",
    alpha: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Calibrate on window 1 across all series and evaluate on window 2 per horizon step (exchangeable across series)."""
    df = fc_df.copy()
    unique_win_ids = sorted(df["window_id"].unique())
    if len(unique_win_ids) >= 2:
        cal_mask = df["window_id"] < unique_win_ids[-1]
    else:
        # Single-window configs (M4): split 50/50 across series so each horizon step h is calibrated across series
        items = sorted(df["item_id"].unique())
        cal_items = set(items[: max(1, len(items) // 2)])
        cal_mask = df["item_id"].isin(cal_items)

    cal_df = df[cal_mask].copy()
    eval_df = df[~cal_mask].copy()
    if eval_df.empty:
        eval_df = cal_df.copy()

    # Normalize scores by per-series scale (window 1 y_true when available, else forecast mean magnitude)
    cal_scales = (
        cal_df.groupby("item_id")["y_true"]
        .apply(lambda s: max(float(np.mean(np.abs(s))), 1e-6))
        .to_dict()
    )
    eval_pred_scales = (
        eval_df.groupby("item_id")["mean"]
        .apply(lambda s: max(float(np.mean(np.abs(s))), 1e-6))
        .to_dict()
    )
    s_cal = cal_df["item_id"].map(lambda i: cal_scales.get(i, 1.0)).to_numpy(dtype=float)
    s_eval = eval_df["item_id"].map(lambda i: cal_scales.get(i, eval_pred_scales.get(i, 1.0))).to_numpy(dtype=float)

    cal_df = cal_df.assign(
        _y=cal_df["y_true"].to_numpy(dtype=float) / s_cal,
        _mean=cal_df["mean"].to_numpy(dtype=float) / s_cal,
        _q10=cal_df["q0.1"].to_numpy(dtype=float) / s_cal,
        _q90=cal_df["q0.9"].to_numpy(dtype=float) / s_cal,
    )
    post_eval_df = eval_df.copy()

    if method == "cqr":
        pooled_off = compute_cqr_offset(
            cal_df["_y"].to_numpy(), cal_df["_q10"].to_numpy(), cal_df["_q90"].to_numpy(), alpha=alpha
        )
        off_by_h = {
            int(h): (
                compute_cqr_offset(g["_y"].to_numpy(), g["_q10"].to_numpy(), g["_q90"].to_numpy(), alpha=alpha)
                if len(g) >= 5
                else pooled_off
            )
            for h, g in cal_df.groupby("horizon_step")
        }
        off_vec = eval_df["horizon_step"].map(lambda h: off_by_h.get(int(h), pooled_off)).to_numpy(dtype=float)
        z_80 = float(norm.ppf(0.9))
        for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS):
            if abs(q - 0.5) < 1e-9:
                continue
            scale_q = abs(float(norm.ppf(q))) / z_80
            sign_q = -1.0 if q < 0.5 else 1.0
            post_eval_df[col] = eval_df[col].to_numpy(dtype=float) + sign_q * scale_q * off_vec * s_eval
        q_mat = np.column_stack([post_eval_df[c].to_numpy(dtype=float) for c in QUANTILE_COLS])
        q_mat = np.maximum.accumulate(q_mat, axis=1)
        for idx, col in enumerate(QUANTILE_COLS):
            post_eval_df[col] = q_mat[:, idx]
        return eval_df, post_eval_df, float(np.mean(list(off_by_h.values())))
    else:
        pooled_hw = compute_abs_residual_conformal_halfwidth(
            cal_df["_y"].to_numpy(), cal_df["_mean"].to_numpy(), alpha=alpha
        )
        hw_by_h = {
            int(h): (
                compute_abs_residual_conformal_halfwidth(g["_y"].to_numpy(), g["_mean"].to_numpy(), alpha=alpha)
                if len(g) >= 5
                else pooled_hw
            )
            for h, g in cal_df.groupby("horizon_step")
        }
        hw_vec = eval_df["horizon_step"].map(lambda h: hw_by_h.get(int(h), pooled_hw)).to_numpy(dtype=float)
        z_80 = float(norm.ppf(0.9))
        sigma_cal = (hw_vec / z_80) * s_eval
        mean_eval = eval_df["mean"].to_numpy(dtype=float)
        for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS):
            post_eval_df[col] = mean_eval + float(norm.ppf(q)) * sigma_cal
        return eval_df, post_eval_df, float(np.mean(list(hw_by_h.values())))


def verify_synthetic_conformal_coverage(seed: int = 20260410) -> Dict[str, float]:
    """Synthetic acceptance test: Verify CQR repairs overconfident intervals to 80% +/- 3% coverage."""
    rng = np.random.default_rng(seed)
    n_cal = 500
    n_eval = 1500
    y_cal = rng.normal(loc=10.0, scale=2.0, size=n_cal)
    y_eval = rng.normal(loc=10.0, scale=2.0, size=n_eval)

    # Deliberately narrow (overconfident) uncalibrated 80% intervals (half-width = 1.0 instead of ~2.56)
    q10_cal, q90_cal = np.full(n_cal, 9.0), np.full(n_cal, 11.0)
    q10_eval, q90_eval = np.full(n_eval, 9.0), np.full(n_eval, 11.0)

    pre_cov = float(np.mean((y_eval >= q10_eval) & (y_eval <= q90_eval)))
    offset = compute_cqr_offset(y_cal, q10_cal, q90_cal, alpha=0.20)
    post_q10 = q10_eval - offset
    post_q90 = q90_eval + offset
    post_cov = float(np.mean((y_eval >= post_q10) & (y_eval <= post_q90)))

    return {
        "pre_coverage_80": pre_cov,
        "post_coverage_80": post_cov,
        "cqr_offset": offset,
        "within_tolerance": float(abs(post_cov - 0.80) <= 0.03),
    }


def run_phase5_uncertainty() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run native-quantile and time-ordered split conformal calibration across all local models & configs."""
    forecasts_root = PROJECT_ROOT / "results" / "forecasts"
    metrics_dir = ensure_dir(PROJECT_ROOT / "results" / "metrics")

    models_to_evaluate = [
        "Chronos_small",
        "Moirai_small",
        "TimesFM",
        "TTM-R2-Pretrained",
        "SeasonalNaive",
        "AutoETS",
        "AutoTheta",
        "AutoARIMA",
        "Naive",
        "Drift",
    ]
    if (forecasts_root / "LightGBM").exists():
        models_to_evaluate.append("LightGBM")

    rows: List[Dict] = []

    for model_id in models_to_evaluate:
        model_dir = forecasts_root / model_id
        if not model_dir.exists():
            continue
        method = "cqr" if model_id in NATIVE_QUANTILE_MODELS else "abs_residual"

        for cfg in LOCAL_CONFIGS:
            safe_cfg = cfg.replace("/", "__")
            pq_path = model_dir / f"{safe_cfg}.parquet"
            if not pq_path.exists():
                continue

            fc_df = pd.read_parquet(pq_path)
            pre_df, post_df, cal_param = apply_conformal_to_forecast_df(
                fc_df, method=method, alpha=0.20
            )

            y_true = pre_df["y_true"].to_numpy(dtype=float)
            s_eval = (
                pre_df.groupby("item_id")["y_true"]
                .transform(lambda s: max(float(np.mean(np.abs(s))), 1e-6))
                .to_numpy(dtype=float)
            )
            y_scaled = y_true / s_eval
            pre_q10_s = pre_df["q0.1"].to_numpy(dtype=float) / s_eval
            pre_q90_s = pre_df["q0.9"].to_numpy(dtype=float) / s_eval
            post_q10_s = post_df["q0.1"].to_numpy(dtype=float) / s_eval
            post_q90_s = post_df["q0.9"].to_numpy(dtype=float) / s_eval

            pre_q_map = {q: pre_df[f"q{q:.1f}"].to_numpy(dtype=float) / s_eval for q in QUANTILE_LEVELS}
            post_q_map = {q: post_df[f"q{q:.1f}"].to_numpy(dtype=float) / s_eval for q in QUANTILE_LEVELS}

            pre_cov_info = coverage_and_width(y_scaled, pre_q10_s, pre_q90_s, y_train=[1.0])
            post_cov_info = coverage_and_width(y_scaled, post_q10_s, post_q90_s, y_train=[1.0])

            pre_crps = crps_weighted_quantile_loss(y_scaled, pre_q_map)
            post_crps = crps_weighted_quantile_loss(y_scaled, post_q_map)

            pre_is = interval_score(y_scaled, pre_q10_s, pre_q90_s, alpha=0.20)
            post_is = interval_score(y_scaled, post_q10_s, post_q90_s, alpha=0.20)

            rows.append(
                {
                    "model": model_id,
                    "config": cfg,
                    "calibration_method": method,
                    "n_eval_points": int(len(y_true)),
                    "calibration_offset": float(cal_param),
                    "pre_coverage_80": float(pre_cov_info["coverage"]),
                    "post_coverage_80": float(post_cov_info["coverage"]),
                    "coverage_error_pre": abs(float(pre_cov_info["coverage"]) - 0.80),
                    "coverage_error_post": abs(float(post_cov_info["coverage"]) - 0.80),
                    "pre_width_80": float(pre_cov_info["scaled_width"]),
                    "post_width_80": float(post_cov_info["scaled_width"]),
                    "pre_crps": float(pre_crps),
                    "post_crps": float(post_crps),
                    "pre_interval_score_80": float(pre_is),
                    "post_interval_score_80": float(post_is),
                }
            )

    cal_df = pd.DataFrame(rows)
    cal_df.to_parquet(metrics_dir / "conformal_calibration.parquet", index=False)

    summary_df = (
        cal_df.groupby(["model", "calibration_method"], as_index=False)
        .agg(
            n_configs=("config", "count"),
            mean_pre_coverage_80=("pre_coverage_80", "mean"),
            mean_post_coverage_80=("post_coverage_80", "mean"),
            mean_abs_coverage_err_pre=("coverage_error_pre", "mean"),
            mean_abs_coverage_err_post=("coverage_error_post", "mean"),
            mean_pre_width_80=("pre_width_80", "mean"),
            mean_post_width_80=("post_width_80", "mean"),
            mean_pre_crps=("pre_crps", "mean"),
            mean_post_crps=("post_crps", "mean"),
            mean_pre_interval_score=("pre_interval_score_80", "mean"),
            mean_post_interval_score=("post_interval_score_80", "mean"),
        )
        .sort_values("mean_post_crps")
        .reset_index(drop=True)
    )
    summary_df.to_csv(metrics_dir / "conformal_summary_by_model.csv", index=False)
    return cal_df, summary_df
