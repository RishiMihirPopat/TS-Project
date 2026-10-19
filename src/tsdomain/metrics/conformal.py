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
    """Split forecast DataFrame into time-ordered cal (earliest 20%) and eval (latest 80%) and calibrate."""
    df = fc_df.copy()
    # Sort strictly by window_id and timestamp/horizon_step to preserve temporal ordering (t_cal < t_eval)
    unique_win_ids = sorted(df["window_id"].unique())
    if len(unique_win_ids) >= 2:
        n_cal_wins = max(1, int(np.floor(0.20 * len(unique_win_ids))))
        cal_wins = set(unique_win_ids[:n_cal_wins])
        cal_mask = df["window_id"].isin(cal_wins)
    else:
        # When w=1 (e.g. M4), split chronologically by timestamp / earliest 20% of series/steps
        df = df.sort_values(["timestamp", "item_id", "horizon_step"]).reset_index(drop=True)
        n_cal = max(1, int(np.floor(0.20 * len(df))))
        cal_mask = pd.Series(False, index=df.index)
        cal_mask.iloc[:n_cal] = True

    cal_df = df[cal_mask].copy()
    eval_df = df[~cal_mask].copy()
    if eval_df.empty:
        eval_df = cal_df.copy()

    y_cal = cal_df["y_true"].to_numpy(dtype=float)
    post_eval_df = eval_df.copy()

    if method == "cqr":
        offset = compute_cqr_offset(
            y_cal,
            cal_df["q0.1"].to_numpy(dtype=float),
            cal_df["q0.9"].to_numpy(dtype=float),
            alpha=alpha,
        )
        z_80 = float(norm.ppf(0.9))
        for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS):
            if abs(q - 0.5) < 1e-9:
                continue
            # Scale CQR offset proportionally across symmetric quantile pairs
            scale_q = abs(float(norm.ppf(q))) / z_80
            sign_q = -1.0 if q < 0.5 else 1.0
            post_eval_df[col] = eval_df[col].to_numpy(dtype=float) + sign_q * scale_q * offset
        # Enforce quantile monotonicity after additive adjustment
        q_mat = np.column_stack([post_eval_df[c].to_numpy(dtype=float) for c in QUANTILE_COLS])
        q_mat = np.maximum.accumulate(q_mat, axis=1)
        for idx, col in enumerate(QUANTILE_COLS):
            post_eval_df[col] = q_mat[:, idx]
        return eval_df, post_eval_df, offset
    else:
        halfwidth_80 = compute_abs_residual_conformal_halfwidth(
            y_cal,
            cal_df["mean"].to_numpy(dtype=float),
            alpha=alpha,
        )
        z_80 = float(norm.ppf(0.9))
        sigma_cal = halfwidth_80 / z_80
        mean_eval = eval_df["mean"].to_numpy(dtype=float)
        for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS):
            post_eval_df[col] = mean_eval + float(norm.ppf(q)) * sigma_cal
        return eval_df, post_eval_df, halfwidth_80


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
            scale_ref = max(float(np.mean(np.abs(y_true))), 1e-6)

            pre_q_map = {q: pre_df[f"q{q:.1f}"].to_numpy(dtype=float) for q in QUANTILE_LEVELS}
            post_q_map = {q: post_df[f"q{q:.1f}"].to_numpy(dtype=float) for q in QUANTILE_LEVELS}

            pre_cov_info = coverage_and_width(
                y_true, pre_df["q0.1"].to_numpy(dtype=float), pre_df["q0.9"].to_numpy(dtype=float)
            )
            post_cov_info = coverage_and_width(
                y_true, post_df["q0.1"].to_numpy(dtype=float), post_df["q0.9"].to_numpy(dtype=float)
            )

            pre_crps = crps_weighted_quantile_loss(y_true, pre_q_map)
            post_crps = crps_weighted_quantile_loss(y_true, post_q_map)

            pre_is = interval_score(
                y_true,
                pre_df["q0.1"].to_numpy(dtype=float),
                pre_df["q0.9"].to_numpy(dtype=float),
                alpha=0.20,
            ) / scale_ref
            post_is = interval_score(
                y_true,
                post_df["q0.1"].to_numpy(dtype=float),
                post_df["q0.9"].to_numpy(dtype=float),
                alpha=0.20,
            ) / scale_ref

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
