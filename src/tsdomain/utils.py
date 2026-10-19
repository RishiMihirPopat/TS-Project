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


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def detect_device(device: str = "auto") -> str:
    """Alias for resolve_device."""
    return resolve_device(device)


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int = 20260410) -> None:
    """Set deterministic random seeds for Python, NumPy, and PyTorch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_forecast_with_metadata(
    df: pd.DataFrame,
    parquet_path: Path,
    model_name: str,
    dataset_config: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save forecast parquet and sidecar metadata with automatic summary fields."""
    comp = (extra_meta or {}).get("compute_summary", {})
    return save_forecast_with_meta(
        df=df,
        out_parquet=parquet_path,
        model_id=model_name,
        package_versions={"torch": "2.14.1+cpu"},
        device=str(comp.get("device", "cpu")),
        seed=20260410,
        context_length=int((extra_meta or {}).get("context_length", 512)),
        n_series=int(df["item_id"].nunique()),
        n_windows=int(df.groupby(["item_id", "window_id"]).ngroups),
        runtime_s=float(comp.get("infer_time_s_total", 0.0)),
        peak_ram_mb=float(comp.get("peak_ram_mb", 0.0)),
        peak_gpu_mb=float(comp.get("peak_vram_mb", 0.0)),
        energy_kwh=float(comp.get("energy_wh_total", 0.0)) / 1000.0,
        energy_method=str(comp.get("energy_source", "tdp_estimate")),
        extra_meta={"dataset_config": dataset_config, **(extra_meta or {})},
    )

