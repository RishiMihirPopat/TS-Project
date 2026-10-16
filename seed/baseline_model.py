"""Phase 0 reference baseline script on the classic Box & Jenkins AirPassengers dataset.

Reproduces the Milestone 1 (Section 3.4) baseline benchmark:
- Dataset: US airline monthly passenger totals (Jan 1949 - Dec 1960, N = 144, m = 12)
- Split: First 120 months train (Jan 1949 - Dec 1958), last 24 months test (Jan 1959 - Dec 1960)
- Models:
  1. ETS (additive damped trend, multiplicative seasonality, period 12)
  2. Theta (deseasonalized Theta method, period 12)
  3. SARIMA(1,1,1)(1,1,1)_12 (unoptimized standard specification)
  4. Seasonal Naive (repeats last observed 12-month cycle)
  5. Drift (linear extrapolation of average training change)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Canonical Box & Jenkins (1976) AirPassengers monthly totals (Jan 1949 - Dec 1960)
AIR_PASSENGERS_VALUES = np.array(
    [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
    ],
    dtype=float,
)


def load_air_passengers() -> pd.Series:
    """Return the monthly AirPassengers series indexed by month-start timestamps."""
    idx = pd.date_range("1949-01-01", periods=len(AIR_PASSENGERS_VALUES), freq="MS")
    return pd.Series(AIR_PASSENGERS_VALUES, index=idx, name="AirPassengers")


def compute_baseline_metrics(
    y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, m: int = 12
) -> Dict[str, float]:
    """Compute MASE, RMSE, and MAPE (%) against in-sample seasonal naive scale."""
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    mase = float(mae / scale)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0)
    return {"MASE": mase, "RMSE": rmse, "MAPE": mape}


def fit_and_forecast_seed_models(
    y_train: pd.Series, horizon: int = 24, season_length: int = 12
) -> Dict[str, np.ndarray]:
    """Fit the 5 reference baseline models on y_train and return h-step forecasts."""
    train_vals = y_train.to_numpy(dtype=float)
    forecasts: Dict[str, np.ndarray] = {}

    # 1. Seasonal Naive: repeat last observed 12-month cycle
    last_cycle = train_vals[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    forecasts["Seasonal Naive"] = np.tile(last_cycle, reps)[:horizon]

    # 2. Drift: linear extrapolation of average change over training period
    slope = (train_vals[-1] - train_vals[0]) / (len(train_vals) - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    forecasts["Drift"] = train_vals[-1] + steps * slope

    # 3. ETS: additive damped trend, multiplicative seasonality (Holt-Winters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ets_model = ExponentialSmoothing(
            y_train,
            trend="add",
            damped_trend=True,
            seasonal="mul",
            seasonal_periods=season_length,
            initialization_method="estimated",
        )
        ets_fit = ets_model.fit(optimized=True)
        forecasts["ETS"] = np.asarray(ets_fit.forecast(horizon), dtype=float)

        # 4. Theta: deseasonalized Theta method, period 12
        theta_model = ThetaModel(
            y_train, period=season_length, deseasonalize=True, method="auto"
        )
        theta_fit = theta_model.fit()
        forecasts["Theta"] = np.asarray(theta_fit.forecast(horizon), dtype=float)

        # 5. SARIMA(1,1,1)(1,1,1)_12: standard unoptimized specification
        sarima_model = ARIMA(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, season_length),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_fit = sarima_model.fit()
        forecasts["SARIMA(1,1,1)(1,1,1)12"] = np.asarray(
            sarima_fit.forecast(horizon), dtype=float
        )

    return forecasts


def run_airpassengers_benchmark(
    test_size: int = 24, season_length: int = 12
) -> pd.DataFrame:
    """Run the AirPassengers baseline evaluation and return sorted results DataFrame."""
    series = load_air_passengers()
    y_train = series.iloc[:-test_size]
    y_test = series.iloc[-test_size:]

    preds = fit_and_forecast_seed_models(
        y_train, horizon=test_size, season_length=season_length
    )
    rows = []
    for model_name, y_hat in preds.items():
        metrics = compute_baseline_metrics(
            y_train.to_numpy(dtype=float),
            y_test.to_numpy(dtype=float),
            y_hat,
            m=season_length,
        )
        rows.append({"Model": model_name, **metrics})

    df = pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
    return df


if __name__ == "__main__":
    results_df = run_airpassengers_benchmark()
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "phase0_airpassengers_seed.csv", index=False)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
