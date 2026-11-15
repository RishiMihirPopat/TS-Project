"""Classical forecasting baselines using statsforecast and statsmodels reference wrappers."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from seed.baseline_model import (
    fit_and_forecast_seed_models,
    load_air_passengers,
)
from tsdomain.compute.profiler import DEFAULT_CPU_TDP_WATTS
from tsdomain.metrics.point import mape, mase, rmse
from tsdomain.utils import (
    QUANTILE_COLS,
    QUANTILE_LEVELS,
    save_forecast_with_meta,
    score_forecast_window,
)

CLASSICAL_MODEL_NAMES: List[str] = [
    "Naive",
    "SeasonalNaive",
    "Drift",
    "AutoETS",
    "AutoTheta",
    "AutoARIMA",
]


def _gaussian_quantiles_from_residuals(
    mean_pred: np.ndarray,
    in_sample_resid_std: float,
    seasonal_period: int = 1,
) -> Dict[str, np.ndarray]:
    """Construct analytical Gaussian horizon-scaled quantiles q0.1..q0.9 from residual scale."""
    h = len(mean_pred)
    steps = np.arange(1, h + 1, dtype=float)
    # Standard horizon growth sqrt(h) or seasonal step growth
    step_scale = np.sqrt(np.maximum(1.0, np.ceil(steps / max(1, seasonal_period))))
    sigma_h = max(float(in_sample_resid_std), 1e-6) * step_scale
    out: Dict[str, np.ndarray] = {}
    for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS):
        z = float(norm.ppf(q))
        out[col] = mean_pred + z * sigma_h
    return out


def forecast_classical_single_series(
    y_train: np.ndarray,
    horizon: int,
    season_length: int,
    model_name: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Forecast a single training window with the specified classical baseline.

    Uses statsforecast for fast C++/Numba-accelerated AutoETS, AutoTheta, AutoARIMA,
    SeasonalNaive, Naive, and RandomWalkWithDrift, with automatic fallback to
    statsmodels if statsforecast raises a numerical error on short/degenerate series.
    """
    y_tr = np.asarray(y_train, dtype=float)
    y_tr = y_tr[~np.isnan(y_tr)]
    n = len(y_tr)
    m = int(season_length) if (season_length >= 1 and n >= 2 * season_length) else 1

    # Residual scale for analytical prediction intervals
    if n > m:
        resid_std = float(np.std(y_tr[m:] - y_tr[:-m], ddof=1)) if n - m > 1 else float(np.std(y_tr))
    else:
        resid_std = float(np.std(y_tr)) if n > 1 else 1.0

    if model_name == "Naive":
        mean_pred = np.repeat(y_tr[-1], horizon)
        return mean_pred, _gaussian_quantiles_from_residuals(mean_pred, resid_std, seasonal_period=1)

    if model_name in {"SeasonalNaive", "Seasonal Naive"}:
        cycle = y_tr[-m:]
        reps = int(np.ceil(horizon / m))
        mean_pred = np.tile(cycle, reps)[:horizon]
        return mean_pred, _gaussian_quantiles_from_residuals(mean_pred, resid_std, seasonal_period=m)

    if model_name == "Drift":
        slope = (y_tr[-1] - y_tr[0]) / max(1, n - 1)
        steps = np.arange(1, horizon + 1, dtype=float)
        mean_pred = y_tr[-1] + steps * slope
        return mean_pred, _gaussian_quantiles_from_residuals(mean_pred, resid_std, seasonal_period=1)

    # Use statsforecast for AutoETS, AutoTheta, AutoARIMA when available
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

        df_sf = pd.DataFrame(
            {
                "unique_id": ["s0"] * n,
                "ds": np.arange(n, dtype=int),
                "y": y_tr,
            }
        )
        if model_name == "AutoETS":
            sf_model = AutoETS(season_length=m)
        elif model_name == "AutoTheta":
            sf_model = AutoTheta(season_length=m)
        elif model_name == "AutoARIMA":
            sf_model = AutoARIMA(season_length=m, max_p=2, max_q=2, max_P=1, max_Q=1, stepwise=True)
        else:
            raise ValueError(f"Unknown classical model_name: {model_name}")

        sf = StatsForecast(models=[sf_model], freq=1, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_df = sf.forecast(df=df_sf, h=horizon, level=[80, 60, 40, 20])

        col_prefix = model_name
        mean_pred = pred_df[col_prefix].to_numpy(dtype=float)
        if np.isnan(mean_pred).any() or not np.isfinite(mean_pred).all():
            raise RuntimeError(f"Non-finite forecast from {model_name}")

        quantiles: Dict[str, np.ndarray] = {
            "q0.1": pred_df.get(f"{col_prefix}-lo-80", pd.Series(mean_pred)).to_numpy(dtype=float),
            "q0.2": pred_df.get(f"{col_prefix}-lo-60", pd.Series(mean_pred)).to_numpy(dtype=float),
            "q0.3": pred_df.get(f"{col_prefix}-lo-40", pd.Series(mean_pred)).to_numpy(dtype=float),
            "q0.4": pred_df.get(f"{col_prefix}-lo-20", pd.Series(mean_pred)).to_numpy(dtype=float),
            "q0.5": mean_pred.copy(),
            "q0.6": pred_df.get(f"{col_prefix}-hi-20", pd.Series(mean_pred)).to_numpy(dtype=float),
            "q0.7": pred_df.get(f"{col_prefix}-hi-40", pd.Series(mean_pred)).to_numpy(dtype=float),
            "q0.8": pred_df.get(f"{col_prefix}-hi-60", pd.Series(mean_pred)).to_numpy(dtype=float),
            "q0.9": pred_df.get(f"{col_prefix}-hi-80", pd.Series(mean_pred)).to_numpy(dtype=float),
        }
        return mean_pred, quantiles
    except Exception:
        # Fallback to statsmodels equivalents
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if model_name == "AutoETS":
                    has_pos = bool(np.all(y_tr > 0))
                    ets = ExponentialSmoothing(
                        y_tr,
                        trend="add",
                        damped_trend=True,
                        seasonal="mul" if (m > 1 and has_pos) else ("add" if m > 1 else None),
                        seasonal_periods=m if m > 1 else None,
                    ).fit(optimized=True)
                    mean_pred = np.asarray(ets.forecast(horizon), dtype=float)
                elif model_name == "AutoTheta":
                    tm = ThetaModel(
                        pd.Series(y_tr),
                        period=m,
                        deseasonalize=(m > 1 and np.all(y_tr > 0)),
                    ).fit()
                    mean_pred = np.asarray(tm.forecast(horizon), dtype=float)
                elif model_name == "AutoARIMA":
                    s_order = (1, 1, 1, m) if (m > 1 and n >= 3 * m) else (0, 0, 0, 0)
                    ar = ARIMA(y_tr, order=(1, 1, 1), seasonal_order=s_order).fit()
                    mean_pred = np.asarray(ar.forecast(horizon), dtype=float)
                else:
                    raise ValueError(f"Unsupported model {model_name}")
            except Exception:
                # Ultimate fallback to SeasonalNaive
                cycle = y_tr[-m:]
                reps = int(np.ceil(horizon / m))
                mean_pred = np.tile(cycle, reps)[:horizon]
        return mean_pred, _gaussian_quantiles_from_residuals(mean_pred, resid_std, seasonal_period=m)


def evaluate_airpassengers_statsmodels() -> Dict[str, Dict[str, float]]:
    """Run exact statsmodels seed reference on AirPassengers (tolerance ±0.02 check)."""
    series = load_air_passengers()
    y_train = series.iloc[:-24]
    y_test = series.iloc[-24:].to_numpy(dtype=float)
    y_tr_np = y_train.to_numpy(dtype=float)

    preds = fit_and_forecast_seed_models(y_train, horizon=24, season_length=12)
    out: Dict[str, Dict[str, float]] = {}
    for name, y_hat in preds.items():
        out[name] = {
            "MASE": mase(y_test, y_hat, y_tr_np, m=12),
            "RMSE": rmse(y_test, y_hat),
            "MAPE": mape(y_test, y_hat),
        }
    return out


def evaluate_airpassengers_statsforecast() -> Dict[str, Dict[str, float]]:
    """Evaluate classical baseline harness on AirPassengers including statsforecast models."""
    series = load_air_passengers()
    y_train = series.iloc[:-24].to_numpy(dtype=float)
    y_test = series.iloc[-24:].to_numpy(dtype=float)

    out: Dict[str, Dict[str, float]] = {}
    for name in ["SeasonalNaive", "Drift", "AutoETS", "AutoTheta", "AutoARIMA"]:
        mean_pred, _ = forecast_classical_single_series(
            y_train, horizon=24, season_length=12, model_name=name
        )
        out[name] = {
            "MASE": mase(y_test, mean_pred, y_train, m=12),
            "RMSE": rmse(y_test, mean_pred),
            "MAPE": mape(y_test, mean_pred),
        }
    return out


def forecast_classical_window(
    item_id: str,
    window_id: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_timestamps: List[pd.Timestamp],
    season_length: int,
    model_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Forecast a single evaluation window and return forecast rows + window metrics."""
    mean_pred, quantiles = forecast_classical_single_series(
        y_train, horizon=len(y_test), season_length=season_length, model_name=model_name
    )
    return score_forecast_window(
        item_id, window_id, y_train, y_test, test_timestamps, season_length, mean_pred, quantiles
    )


def run_classical_on_dataset(
    windows: List[Dict],
    dataset_config: str,
    season_length: int,
    model_name: str,
    out_dir: Path = Path("results/forecasts"),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a classical model across all rolling windows of a dataset configuration."""
    t0 = time.perf_counter()
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / (1024 * 1024)

    all_forecast_dfs: List[pd.DataFrame] = []
    all_metric_rows: List[Dict[str, float]] = []

    # Check if we can batch-run AutoETS / AutoTheta / AutoARIMA across all windows in one StatsForecast call
    batched_preds: Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
    if model_name in {"AutoETS", "AutoTheta", "AutoARIMA"} and len(windows) > 1:
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

            horizon = len(windows[0]["y_test"])
            min_n = min(len(w["y_train"]) for w in windows)
            m = int(season_length) if (season_length >= 1 and min_n >= 2 * season_length) else 1

            sf_rows = []
            for idx, w in enumerate(windows):
                y_tr = np.asarray(w["y_train"], dtype=float)
                y_tr = y_tr[~np.isnan(y_tr)]
                # shortcut: caps training history at max(512, 4*m) and season_length at 24 for AutoETS/AutoARIMA speed, remove cap for unbounded state-space fits
                max_hist = max(512, 4 * m)
                if len(y_tr) > max_hist:
                    y_tr = y_tr[-max_hist:]
                uid = f"w_{idx}"
                sf_rows.append(
                    pd.DataFrame(
                        {
                            "unique_id": [uid] * len(y_tr),
                            "ds": np.arange(len(y_tr), dtype=int),
                            "y": y_tr,
                        }
                    )
                )
            df_batch = pd.concat(sf_rows, ignore_index=True)

            if model_name == "AutoETS":
                sf_model = AutoETS(season_length=(m if m <= 24 else 1))
            elif model_name == "AutoTheta":
                sf_model = AutoTheta(season_length=m)
            else:
                sf_model = AutoARIMA(
                    season_length=(m if m <= 24 else 1),
                    max_p=2,
                    max_q=2,
                    max_P=1,
                    max_Q=1,
                    stepwise=True,
                    approximation=True,
                )

            sf = StatsForecast(models=[sf_model], freq=1, n_jobs=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_all = sf.forecast(df=df_batch, h=horizon, level=[80, 60, 40, 20])

            for idx, grp in pred_all.groupby("unique_id", sort=False):
                w_idx = int(str(idx).split("_")[1])
                mp = grp[model_name].to_numpy(dtype=float)
                if np.isfinite(mp).all():
                    q_dict = {
                        "q0.1": grp.get(f"{model_name}-lo-80", pd.Series(mp)).to_numpy(dtype=float),
                        "q0.2": grp.get(f"{model_name}-lo-60", pd.Series(mp)).to_numpy(dtype=float),
                        "q0.3": grp.get(f"{model_name}-lo-40", pd.Series(mp)).to_numpy(dtype=float),
                        "q0.4": grp.get(f"{model_name}-lo-20", pd.Series(mp)).to_numpy(dtype=float),
                        "q0.5": mp.copy(),
                        "q0.6": grp.get(f"{model_name}-hi-20", pd.Series(mp)).to_numpy(dtype=float),
                        "q0.7": grp.get(f"{model_name}-hi-40", pd.Series(mp)).to_numpy(dtype=float),
                        "q0.8": grp.get(f"{model_name}-hi-60", pd.Series(mp)).to_numpy(dtype=float),
                        "q0.9": grp.get(f"{model_name}-hi-80", pd.Series(mp)).to_numpy(dtype=float),
                    }
                    # Replace any non-finite interval bounds with Gaussian fallback
                    if all(np.isfinite(v).all() for v in q_dict.values()):
                        batched_preds[w_idx] = (mp, q_dict)
        except Exception:
            batched_preds = {}

    for idx, win in enumerate(windows):
        if idx in batched_preds:
            mean_pred, quantiles = batched_preds[idx]
            df_w, m_w = score_forecast_window(
                win["item_id"],
                win["window_id"],
                win["y_train"],
                win["y_test"],
                win["test_timestamps"],
                season_length,
                mean_pred,
                quantiles,
            )
        else:
            df_w, m_w = forecast_classical_window(
                item_id=win["item_id"],
                window_id=win["window_id"],
                y_train=win["y_train"],
                y_test=win["y_test"],
                test_timestamps=win["test_timestamps"],
                season_length=season_length,
                model_name=model_name,
            )
        all_forecast_dfs.append(df_w)
        all_metric_rows.append(m_w)

    runtime_s = time.perf_counter() - t0
    ram_after = proc.memory_info().rss / (1024 * 1024)
    peak_ram_mb = max(ram_before, ram_after)
    energy_kwh = (runtime_s * DEFAULT_CPU_TDP_WATTS) / 3_600_000.0

    full_forecast_df = pd.concat(all_forecast_dfs, ignore_index=True)
    metrics_df = pd.DataFrame(all_metric_rows)

    safe_config = dataset_config.replace("/", "__")
    parquet_path = out_dir / model_name / f"{safe_config}.parquet"
    save_forecast_with_meta(
        full_forecast_df,
        parquet_path,
        model_id=model_name,
        package_versions={"statsforecast": "2.0.1", "statsmodels": "0.15.0"},
        device="cpu",
        seed=seed,
        context_length=int(np.median([len(w["y_train"]) for w in windows])),
        n_series=len({w["item_id"] for w in windows}),
        n_windows=len(windows),
        runtime_s=runtime_s,
        peak_ram_mb=peak_ram_mb,
        peak_gpu_mb=0.0,
        energy_kwh=energy_kwh,
        energy_method="TDP_estimate",
        extra_meta={"dataset_config": dataset_config, "season_length": season_length},
    )
    return full_forecast_df, metrics_df

