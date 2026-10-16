"""Phase 0 acceptance tests on AirPassengers (Section 3.4 & Phase 0 acceptance checks)."""

from __future__ import annotations

import pytest

from tsdomain.baselines.classical import (
    evaluate_airpassengers_statsforecast,
    evaluate_airpassengers_statsmodels,
)


def test_airpassengers_statsmodels_reference_strict() -> None:
    """Acceptance check: statsmodels reference matches Section 3.4 numbers within ±0.02."""
    res = evaluate_airpassengers_statsmodels()

    assert res["ETS"]["MASE"] == pytest.approx(1.278, abs=0.02)
    assert res["Theta"]["MASE"] == pytest.approx(2.213, abs=0.02)
    assert res["SARIMA(1,1,1)(1,1,1)12"]["MASE"] == pytest.approx(2.446, abs=0.02)
    assert res["Seasonal Naive"]["MASE"] == pytest.approx(2.494, abs=0.02)
    assert res["Drift"]["MASE"] == pytest.approx(3.206, abs=0.02)

    # Also verify RMSE and MAPE match Section 3.4 table
    assert res["ETS"]["RMSE"] == pytest.approx(41.00, abs=0.5)
    assert res["ETS"]["MAPE"] == pytest.approx(8.01, abs=0.1)


def test_airpassengers_statsforecast_harness() -> None:
    """Acceptance check: statsforecast refactored harness runs cleanly on AirPassengers."""
    res = evaluate_airpassengers_statsforecast()
    assert res["SeasonalNaive"]["MASE"] == pytest.approx(2.494, abs=0.02)
    assert res["Drift"]["MASE"] == pytest.approx(3.206, abs=0.02)
    # statsforecast AutoTheta selects Dynamic Optimized Theta with multiplicative seasonal decomposition (~1.792)
    assert 0.5 < res["AutoTheta"]["MASE"] < 2.30
    assert 0.5 < res["AutoETS"]["MASE"] < 2.20
    assert 0.5 < res["AutoARIMA"]["MASE"] < 2.60

