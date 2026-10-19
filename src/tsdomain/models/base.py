"""Phase 4: Common Preprocessing/Scaling Protocol and Base Adapter for Zero-Shot TSFMs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from tsdomain.metrics.point import mape, mase, rmse, rmsse
from tsdomain.metrics.probabilistic import (
    coverage_and_width,
    crps_weighted_quantile_loss,
    interval_score,
)
from tsdomain.utils import QUANTILE_COLS, QUANTILE_LEVELS


def apply_common_scaling(
    y_train: np.ndarray,
    context_length: int = 512,
    mode: str = "common_zscore",
) -> Tuple[np.ndarray, float, float]:
    """Truncate to `context_length` and apply the common z-score scaling protocol.

    If `mode == "model_default"`, returns the raw truncated context with (loc=0.0, scale=1.0)
    so each model's internal instance normalization operates directly on raw units.
    """
    arr = np.asarray(y_train, dtype=float)
    arr = arr[~np.isnan(arr)]
    ctx = arr[-context_length:] if len(arr) > context_length else arr.copy()

    if mode == "model_default":
        return ctx, 0.0, 1.0

    loc = float(np.mean(ctx))
    scale = float(np.std(ctx, ddof=1)) if len(ctx) > 1 else 1.0
    if scale < 1e-6 or not np.isfinite(scale):
        scale = 1.0
    z_ctx = (ctx - loc) / scale
    return z_ctx, loc, scale


def invert_common_scaling(
    mean_pred: np.ndarray,
    quantiles_pred: Dict[str, np.ndarray],
    loc: float,
    scale: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Invert the common z-score scaling back to original series units."""
    mean_unscaled = np.asarray(mean_pred, dtype=float) * scale + loc
    q_unscaled = {
        col: np.asarray(q_arr, dtype=float) * scale + loc
        for col, q_arr in quantiles_pred.items()
    }
    return mean_unscaled, q_unscaled


def gaussian_quantiles_from_context(
    mean_pred: np.ndarray,
    y_context: np.ndarray,
    season_length: int = 1,
) -> Dict[str, np.ndarray]:
    """Construct analytical quantiles q0.1..q0.9 for models that emit point forecasts only."""
    h = len(mean_pred)
    n = len(y_context)
    m = int(season_length) if (season_length >= 1 and n >= 2 * season_length) else 1
    if n > m:
        resid_std = float(np.std(y_context[m:] - y_context[:-m], ddof=1))
    else:
        resid_std = float(np.std(y_context, ddof=1)) if n > 1 else 1.0
    resid_std = max(resid_std, 1e-6)
    steps = np.arange(1, h + 1, dtype=float)
    step_scale = np.sqrt(np.maximum(1.0, np.ceil(steps / max(1, m))))
    sigma_h = resid_std * step_scale
    return {
        col: mean_pred + float(norm.ppf(q)) * sigma_h
        for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS)
    }


class TSFMBaseAdapter(ABC):
    """Abstract base class for zero-shot Time-Series Foundation Model adapters."""

    model_id: str
    checkpoint_id: str
    param_count: int
    device: str
    context_length: int

    @abstractmethod
    def load_model(self) -> float:
        """Load pretrained weights onto `self.device` and return load wall-clock seconds."""

    @abstractmethod
    def predict_batch(
        self,
        contexts: List[np.ndarray],
        horizon: int,
        season_length: int = 1,
    ) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Run zero-shot batch inference and return list of (mean_pred, quantiles_dict) per context."""

    def forecast_windows(
        self,
        windows: List[Dict],
        scaling_mode: str = "common_zscore",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Forecast all rolling windows of a configuration under the common protocol."""
        if not windows:
            raise ValueError("windows list must not be empty.")

        horizon = int(windows[0]["prediction_length"])
        season_length = int(windows[0]["season_length"])

        scaled_contexts: List[np.ndarray] = []
        locs: List[float] = []
        scales: List[float] = []

        for w in windows:
            z_ctx, loc, sc = apply_common_scaling(
                w["y_train"], context_length=self.context_length, mode=scaling_mode
            )
            scaled_contexts.append(z_ctx)
            locs.append(loc)
            scales.append(sc)

        batch_preds = self.predict_batch(
            scaled_contexts, horizon=horizon, season_length=season_length
        )

        forecast_dfs: List[pd.DataFrame] = []
        metric_rows: List[Dict[str, float]] = []

        for w, (mean_z, q_z), loc, sc in zip(windows, batch_preds, locs, scales):
            mean_pred, quantiles = invert_common_scaling(mean_z, q_z, loc, sc)
            y_train = w["y_train"]
            y_test = w["y_test"]
            test_ts = w["test_timestamps"]
            item_id = str(w["item_id"])
            win_id = int(w["window_id"])

            rows = []
            for step_idx in range(horizon):
                r = {
                    "item_id": item_id,
                    "window_id": win_id,
                    "horizon_step": int(step_idx + 1),
                    "timestamp": pd.Timestamp(test_ts[step_idx]),
                    "y_true": float(y_test[step_idx]),
                    "mean": float(mean_pred[step_idx]),
                }
                for col in QUANTILE_COLS:
                    r[col] = float(quantiles[col][step_idx])
                rows.append(r)
            forecast_dfs.append(pd.DataFrame(rows))

            q_map = {q: quantiles[f"q{q:.1f}"] for q in QUANTILE_LEVELS}
            cov_info = coverage_and_width(
                y_test, quantiles["q0.1"], quantiles["q0.9"], y_train=y_train
            )
            metric_rows.append(
                {
                    "item_id": item_id,
                    "window_id": win_id,
                    "mase": mase(y_test, mean_pred, y_train, m=season_length),
                    "rmsse": rmsse(y_test, mean_pred, y_train, m=season_length),
                    "rmse": rmse(y_test, mean_pred),
                    "mape": mape(y_test, mean_pred),
                    "crps": crps_weighted_quantile_loss(y_test, q_map),
                    "coverage_80": cov_info["coverage"],
                    "width_80": cov_info["scaled_width"],
                    "interval_score_80": interval_score(
                        y_test, quantiles["q0.1"], quantiles["q0.9"], alpha=0.2
                    ),
                }
            )

        return pd.concat(forecast_dfs, ignore_index=True), pd.DataFrame(metric_rows)
