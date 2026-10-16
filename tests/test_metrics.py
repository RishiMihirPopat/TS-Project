"""Unit tests for Section 5 point and probabilistic forecasting metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tsdomain.metrics.point import (
    geometric_mean_ratios,
    log_ratio,
    mape,
    mase,
    qlike,
    rmse,
    rmsse,
)
from tsdomain.metrics.probabilistic import (
    coverage_and_width,
    crps_weighted_quantile_loss,
    interval_score,
    pinball_loss,
)


def test_mase_hand_computed() -> None:
    """Verify MASE against a hand-computed example with season length m=2."""
    # y_train = [10, 20, 12, 22, 14, 24] -> seasonal diffs (m=2): |12-10|=2, |22-20|=2, |14-12|=2, |24-22|=2
    # Denominator = 2.0
    y_train = np.array([10.0, 20.0, 12.0, 22.0, 14.0, 24.0])
    y_true = np.array([16.0, 26.0])
    y_pred = np.array([15.0, 23.0])  # errors: |16-15|=1, |26-23|=3 -> MAE = 2.0
    # MASE = 2.0 / 2.0 = 1.0
    assert mase(y_true, y_pred, y_train, m=2) == pytest.approx(1.0, abs=1e-9)


def test_mase_short_series_fallback_to_m1() -> None:
    """Verify MASE falls back to m=1 when len(y_train) < 2*m."""
    y_train = np.array([1.0, 3.0, 5.0, 7.0])  # len=4 < 2*12; m=1 diffs are all 2.0
    y_true = np.array([9.0, 11.0])
    y_pred = np.array([8.0, 10.0])  # MAE = 1.0 -> MASE = 1.0 / 2.0 = 0.5
    assert mase(y_true, y_pred, y_train, m=12) == pytest.approx(0.5, abs=1e-9)


def test_rmsse_hand_computed() -> None:
    """Verify RMSSE against a hand-computed example."""
    y_train = np.array([10.0, 20.0, 12.0, 22.0])  # m=2 diffs: 2, 2 -> mean sq diff = 4.0
    y_true = np.array([14.0, 24.0])
    y_pred = np.array([10.0, 20.0])  # sq errors: 16, 16 -> mean sq error = 16.0
    # RMSSE = sqrt(16.0 / 4.0) = 2.0
    assert rmsse(y_true, y_pred, y_train, m=2) == pytest.approx(2.0, abs=1e-9)


def test_pinball_and_crps() -> None:
    """Verify pinball loss and weighted quantile loss CRPS approximation."""
    y_true = np.array([10.0, 20.0])
    q_pred = np.array([8.0, 22.0])
    # tau = 0.5: max(0.5*(10-8), -0.5*(10-8)) = 1.0; max(0.5*(20-22), -0.5*(20-22)) = 1.0 -> mean = 1.0
    assert pinball_loss(y_true, q_pred, tau=0.5) == pytest.approx(1.0, abs=1e-9)

    # Perfect quantile predictions should give CRPS = 0
    perfect_q = {q: y_true for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    assert crps_weighted_quantile_loss(y_true, perfect_q) == pytest.approx(0.0, abs=1e-12)


def test_coverage_and_interval_score() -> None:
    """Verify empirical coverage and Winkler interval score."""
    y_true = np.array([5.0, 10.0, 15.0, 25.0])
    q_low = np.array([4.0, 8.0, 12.0, 18.0])
    q_high = np.array([6.0, 12.0, 18.0, 22.0])
    # First 3 inside, 4th (25.0 > 22.0) outside -> coverage = 3/4 = 0.75
    cov = coverage_and_width(y_true, q_low, q_high)
    assert cov["coverage"] == pytest.approx(0.75, abs=1e-9)
    assert cov["raw_width"] == pytest.approx((2.0 + 4.0 + 6.0 + 4.0) / 4.0, abs=1e-9)

    # Interval score at alpha=0.2:
    # widths = [2, 4, 6, 4], penalty on 4th = (2/0.2)*(25 - 22) = 10 * 3 = 30
    # mean IS = (2 + 4 + 6 + (4 + 30)) / 4 = 46 / 4 = 11.5
    assert interval_score(y_true, q_low, q_high, alpha=0.2) == pytest.approx(11.5, abs=1e-9)


def test_qlike_and_log_ratio() -> None:
    """Verify QLIKE loss and log_ratio."""
    var_true = np.array([0.04, 0.09])
    # At var_pred == var_true, QLIKE = 1 - log(1) - 1 = 0
    assert qlike(var_true, var_true) == pytest.approx(0.0, abs=1e-12)
    assert qlike(var_true, var_true * 2.0) > 0.0

    # log_ratio: model MASE = 0.8, classical = 1.0 -> negative (TSFM wins)
    assert log_ratio(0.8, 1.0) == pytest.approx(np.log(0.8), abs=1e-9)
    assert geometric_mean_ratios([0.5, 2.0]) == pytest.approx(1.0, abs=1e-9)
    assert rmse([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)
    assert mape([100.0, 200.0], [90.0, 220.0]) == pytest.approx(10.0, abs=1e-9)
