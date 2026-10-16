"""Classical and gradient-boosted baseline forecasters."""

from tsdomain.baselines.classical import (
    CLASSICAL_MODEL_NAMES,
    evaluate_airpassengers_statsforecast,
    evaluate_airpassengers_statsmodels,
    forecast_classical_window,
    run_classical_on_dataset,
)

__all__ = [
    "CLASSICAL_MODEL_NAMES",
    "evaluate_airpassengers_statsmodels",
    "evaluate_airpassengers_statsforecast",
    "forecast_classical_window",
    "run_classical_on_dataset",
]
