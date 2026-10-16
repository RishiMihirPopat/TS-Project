"""Probabilistic forecasting metrics implementing Section 5 of the Build Plan."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np


def pinball_loss(y_true: Iterable[float], q_pred: Iterable[float], tau: float) -> float:
    """Mean pinball (quantile) loss at quantile level tau in (0, 1).

    pinball_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q))
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"Quantile level tau must be in (0, 1), got {tau}.")
    yt = np.asarray(list(y_true), dtype=float)
    qp = np.asarray(list(q_pred), dtype=float)
    diff = yt - qp
    loss = np.maximum(tau * diff, (tau - 1.0) * diff)
    return float(np.mean(loss))


def crps_weighted_quantile_loss(
    y_true: Iterable[float],
    quantiles: Mapping[float, Iterable[float]],
    eps: float = 1e-12,
) -> float:
    """Weighted quantile loss approximation to CRPS matching GIFT-Eval (Section 5).

    CRPS = (2 / |Q|) * sum_{q in Q} (sum_t pinball_q(y_t, q_hat_t)) / (sum_t |y_t|)
    """
    if not quantiles:
        raise ValueError("quantiles mapping must not be empty.")
    yt = np.asarray(list(y_true), dtype=float)
    denom = max(float(np.sum(np.abs(yt))), eps)
    total_weighted_pinball = 0.0
    for tau, q_vals in quantiles.items():
        qp = np.asarray(list(q_vals), dtype=float)
        diff = yt - qp
        pb_sum = float(np.sum(np.maximum(float(tau) * diff, (float(tau) - 1.0) * diff)))
        total_weighted_pinball += pb_sum / denom
    return float((2.0 / len(quantiles)) * total_weighted_pinball)


def coverage_and_width(
    y_true: Iterable[float],
    q_low: Iterable[float],
    q_high: Iterable[float],
    y_train: Optional[Iterable[float]] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute empirical interval coverage and scale-normalized interval width (Section 5)."""
    yt = np.asarray(list(y_true), dtype=float)
    ql = np.asarray(list(q_low), dtype=float)
    qh = np.asarray(list(q_high), dtype=float)
    if np.any(qh < ql):
        # Guard against quantile crossing if uncalibrated
        ql_clean = np.minimum(ql, qh)
        qh_clean = np.maximum(ql, qh)
    else:
        ql_clean, qh_clean = ql, qh

    inside = (yt >= ql_clean) & (yt <= qh_clean)
    coverage = float(np.mean(inside))
    raw_width = float(np.mean(qh_clean - ql_clean))
    if y_train is not None:
        ytr = np.asarray(list(y_train), dtype=float)
        scale = max(float(np.mean(np.abs(ytr[~np.isnan(ytr)]))), eps)
    else:
        scale = max(float(np.mean(np.abs(yt))), eps)
    return {
        "coverage": coverage,
        "raw_width": raw_width,
        "scaled_width": float(raw_width / scale),
    }


def interval_score(
    y_true: Iterable[float],
    q_low: Iterable[float],
    q_high: Iterable[float],
    alpha: float = 0.2,
) -> float:
    """Winkler Interval Score at nominal (1 - alpha) coverage (Section 5).

    IS_alpha = (u - l) + (2/alpha)*(l - y)*1[y < l] + (2/alpha)*(y - u)*1[y > u]
    For [q0.1, q0.9], nominal coverage is 80%, so alpha = 0.2.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    yt = np.asarray(list(y_true), dtype=float)
    ql = np.asarray(list(q_low), dtype=float)
    qh = np.asarray(list(q_high), dtype=float)
    l = np.minimum(ql, qh)
    u = np.maximum(ql, qh)

    width = u - l
    below = np.where(yt < l, (2.0 / alpha) * (l - yt), 0.0)
    above = np.where(yt > u, (2.0 / alpha) * (yt - u), 0.0)
    return float(np.mean(width + below + above))
