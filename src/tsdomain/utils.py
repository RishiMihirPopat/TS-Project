"""Shared utilities for forecast parquet schema validation, metadata sidecars, and device resolution."""

from __future__ import annotations

import datetime as dt
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

QUANTILE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILE_COLS: List[str] = [f"q{q:.1f}" for q in QUANTILE_LEVELS]

REQUIRED_FORECAST_COLUMNS: List[str] = [
    "item_id",
    "window_id",
    "horizon_step",
    "timestamp",
    "y_true",
    "mean",
    *QUANTILE_COLS,
]


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto', 'cpu', or 'cuda' against runtime availability."""
    device = device.lower().strip()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device}'. Expected one of {{auto, cpu, cuda}}.")
    if device == "cpu":
        return "cpu"
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if device == "cuda" and not has_cuda:
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    return "cuda" if has_cuda else "cpu"


def validate_forecast_dataframe(df: pd.DataFrame) -> None:
    """Validate that a forecast DataFrame conforms to the Section 3 Forecast Parquet Schema."""
    missing = [col for col in REQUIRED_FORECAST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Forecast DataFrame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Forecast DataFrame must not be empty.")
    if df["y_true"].isna().any():
        raise ValueError("Forecast DataFrame contains NaN values in y_true.")
    if df["mean"].isna().any():
        raise ValueError("Forecast DataFrame contains NaN values in mean.")
    if (df["horizon_step"] < 1).any():
        raise ValueError("horizon_step must be 1-indexed (>= 1).")


def save_forecast_with_meta(
    df: pd.DataFrame,
    out_parquet: Path,
    *,
    model_id: str,
    package_versions: Dict[str, str],
    device: str,
    seed: int,
    context_length: int,
    n_series: int,
    n_windows: int,
    runtime_s: float,
    peak_ram_mb: float,
    peak_gpu_mb: float,
    energy_kwh: float,
    energy_method: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Validate and save forecast parquet along with its sidecar .meta.json."""
    validate_forecast_dataframe(df)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    # Ensure canonical column ordering
    extra_cols = [c for c in df.columns if c not in REQUIRED_FORECAST_COLUMNS]
    ordered_df = df[REQUIRED_FORECAST_COLUMNS + extra_cols].copy()
    ordered_df.to_parquet(out_parquet, index=False)

    meta_path = out_parquet.with_suffix(".parquet.meta.json")
    meta_payload: Dict[str, Any] = {
        "model_id": model_id,
        "package_versions": package_versions,
        "device": resolve_device(device) if device in {"auto", "cpu", "cuda"} else device,
        "seed": int(seed),
        "context_length": int(context_length),
        "n_series": int(n_series),
        "n_windows": int(n_windows),
        "runtime_s": float(runtime_s),
        "peak_ram_mb": float(peak_ram_mb),
        "peak_gpu_mb": float(peak_gpu_mb),
        "energy_kwh": float(energy_kwh),
        "energy_method": str(energy_method),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
    }
    if extra_meta:
        meta_payload.update(extra_meta)
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return meta_path
