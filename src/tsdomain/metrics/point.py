"""Point forecasting metrics implementing Section 5 of the Build Plan."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _seasonal_scale_denom(y_train: np.ndarray, m: int, power: int = 1) -> float:
    """Compute in-sample seasonal naive denominator with fallback to m=1 if len < 2*m."""
    y_tr = np.asarray(y_train, dtype=float)
    y_tr = y_tr[~np.isnan(y_tr)]
    eff_m = int(m) if (m >= 1 and len(y_tr) >= 2 * m) else 1
    if len(y_tr) <= eff_m:
        raise ValueError(f"Training series length ({len(y_tr)}) must exceed period ({eff_m}).")
    diffs = np.abs(y_tr[eff_m:] - y_tr[:-eff_m])
    if power == 1:
        denom = float(np.mean(diffs))
    elif power == 2:
        denom = float(np.mean(diffs ** 2))
    else:
        raise ValueError(f"Unsupported power={power}")
    if denom <= 1e-12:
        # Constant training series fallback: scale by std or 1.0 to avoid div-by-zero
        fallback = float(np.mean(np.abs(y_tr))) if np.mean(np.abs(y_tr)) > 1e-12 else 1.0
        return fallback
    return denom


def mase(y_true: Iterable[float], y_pred: Iterable[float], y_train: Iterable[float], m: int = 1) -> float:
    """Mean Absolute Scaled Error (Section 5).

    MASE = mean_t |y_t - y_hat_t| / mean_s |y_s - y_{s-m}|
    Falls back to m = 1 if len(y_train) < 2*m.
    """
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    ytr = np.asarray(list(y_train), dtype=float)
    denom = _seasonal_scale_denom(ytr, m=m, power=1)
    num = float(np.mean(np.abs(yt - yp)))
    return float(num / denom)


def rmsse(y_true: Iterable[float], y_pred: Iterable[float], y_train: Iterable[float], m: int = 1) -> float:
    """Root Mean Squared Scaled Error (Section 5).

    RMSSE = sqrt(mean_t (y_t - y_hat_t)^2 / mean_s (y_s - y_{s-m})^2)
    """
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    ytr = np.asarray(list(y_train), dtype=float)
    denom = _seasonal_scale_denom(ytr, m=m, power=2)
    num = float(np.mean((yt - yp) ** 2))
    return float(np.sqrt(num / denom))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-12) -> float:
    """Mean Absolute Percentage Error in percent (0-100 scale)."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def qlike(var_true: Iterable[float], var_pred: Iterable[float], eps: float = 1e-12) -> float:
    """Patton (2011) QLIKE loss on variance forecasts (Section 5).

    QLIKE = mean_t (sigma^2_true / sigma^2_hat - log(sigma^2_true / sigma^2_hat) - 1)
    """
    vt = np.maximum(np.asarray(list(var_true), dtype=float), eps)
    vp = np.maximum(np.asarray(list(var_pred), dtype=float), eps)
    ratio = vt / vp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def log_ratio(mase_model: float, mase_best_classical: float, eps: float = 1e-12) -> float:
    """Compute log(MASE_model / MASE_best_classical). Negative means TSFM wins."""
    m_mod = max(float(mase_model), eps)
    m_cls = max(float(mase_best_classical), eps)
    return float(np.log(m_mod / m_cls))


def aggregate_config_metrics(per_window_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate per-series, per-window metrics to configuration level via arithmetic mean."""
    numeric_cols = [
        c for c in ["mase", "rmsse", "rmse", "mape", "crps", "coverage_80", "width_80", "interval_score_80"]
        if c in per_window_df.columns
    ]
    return {col: float(per_window_df[col].mean()) for col in numeric_cols}


def geometric_mean_ratios(ratios: Iterable[float], eps: float = 1e-12) -> float:
    """Aggregate ratios across configs by the geometric mean: exp(mean(log(r)))."""
    arr = np.maximum(np.asarray(list(ratios), dtype=float), eps)
    return float(np.exp(np.mean(np.log(arr))))
