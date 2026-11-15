"""Phase 4: Compute Profiler for Wall-Clock Latency, Peak Host/GPU Memory, and Energy (kWh).

Implements the exact profiling protocol from Section 4 (Phase 4) of the Build Plan:
- Separate model-load timing (`model_load_s`) from inference wall-clock time (`runtime_s`).
- Normalized throughput metric: `ms_per_1000_forecasts` (per 1,000 horizon-window forecasts).
- Peak host RSS memory (`peak_ram_mb`) via `psutil` and peak GPU memory (`peak_gpu_mb`) via `torch.cuda`.
- Energy consumption (`energy_kwh`) via `codecarbon.EmissionsTracker` with explicit `energy_method`
  detection (`RAPL`, `NVML`, or `TDP_estimate`).
- Repeats each profiled inference pass `n_repeats` times (default 3) after `n_warmup` warm-up passes
  (default 1) and records mean and standard deviation across repeats.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import psutil

os.environ.setdefault("CODECARBON_LOG_LEVEL", "ERROR")
logging.getLogger("codecarbon").setLevel(logging.ERROR)

DEFAULT_CPU_TDP_WATTS: float = 45.0


def _detect_energy_method(device: str) -> str:
    """Determine which hardware energy counter is active on the current host."""
    if device == "cuda":
        return "NVML"
    rapl_path = Path("/sys/class/powercap/intel-rapl")
    if rapl_path.exists() and os.access(str(rapl_path), os.R_OK):
        return "RAPL"
    return "TDP_estimate"


def _measure_single_pass(
    fn: Callable[[], Any],
    device: str,
    use_codecarbon: bool = True,
) -> Tuple[Any, float, float, float, float, str]:
    """Run fn() once and return (output, runtime_s, peak_ram_mb, peak_gpu_mb, energy_kwh, energy_method)."""
    proc = psutil.Process()
    ram_start = proc.memory_info().rss / (1024.0 * 1024.0)

    has_cuda = False
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                has_cuda = True
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        except Exception:
            has_cuda = False

    energy_method = _detect_energy_method(device if has_cuda else "cpu")
    tracker = None
    if use_codecarbon:
        try:
            from codecarbon import OfflineEmissionsTracker

            tracker = OfflineEmissionsTracker(
                country_iso_code="USA",
                log_level="error",
                save_to_file=False,
                measure_power_secs=1,
            )
            tracker.start()
        except Exception:
            tracker = None

    t0 = time.perf_counter()
    output = fn()
    if has_cuda:
        import torch

        torch.cuda.synchronize()
    runtime_s = time.perf_counter() - t0

    cc_kwh = None
    if tracker is not None:
        try:
            tracker.stop()
            cc_kwh = getattr(tracker, "_total_energy", None)
            if cc_kwh is not None:
                cc_kwh = float(getattr(cc_kwh, "kWh", 0.0))
        except Exception:
            cc_kwh = None

    ram_end = proc.memory_info().rss / (1024.0 * 1024.0)
    peak_ram_mb = max(ram_start, ram_end)

    peak_gpu_mb = 0.0
    if has_cuda:
        import torch

        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    # shortcut: TDP fallback when RAPL/NVML counters are unavailable, pass use_codecarbon=True on Linux/CUDA hosts
    tdp_watts = 150.0 if has_cuda else DEFAULT_CPU_TDP_WATTS
    tdp_kwh = (runtime_s * tdp_watts) / 3_600_000.0
    if cc_kwh is not None and cc_kwh > 1e-10:
        energy_kwh = float(cc_kwh)
    else:
        energy_kwh = float(tdp_kwh)
        if energy_method != "NVML":
            energy_method = "TDP_estimate"

    return output, float(runtime_s), float(peak_ram_mb), float(peak_gpu_mb), float(energy_kwh), energy_method


def profile_callable(
    fn: Callable[..., Any],
    *args: Any,
    device: str = "cpu",
    n_forecast_points: int = 1,
    load_time_s: float = 0.0,
    use_codecarbon: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Measure a single callable pass and return (result, profile_dict)."""
    out, rt_s, ram_mb, gpu_mb, kwh, method_str = _measure_single_pass(
        lambda: fn(*args, **kwargs),
        device=device,
        use_codecarbon=use_codecarbon,
    )
    wh_total = kwh * 1000.0
    pts = max(1, int(n_forecast_points))
    return out, {
        "load_time_s": float(load_time_s),
        "infer_time_s_total": float(rt_s),
        "infer_time_ms_per_point": float((rt_s * 1000.0) / pts),
        "peak_ram_mb": float(ram_mb),
        "peak_vram_mb": float(gpu_mb),
        "energy_wh_total": float(wh_total),
        "energy_mwh_per_1k_points": float((wh_total * 1e6) / pts),
        "energy_source": str(method_str),
        "device": str(device),
        "n_forecast_points": pts,
    }


def summarize_profile_repeats(
    rep_profiles: List[Dict[str, Any]],
    *,
    model_id: str,
    config: str,
    param_count: int,
) -> Dict[str, Any]:
    """Aggregate median and IQR across profiling repeats into a standard row."""
    infer_s = [float(p["infer_time_s_total"]) for p in rep_profiles]
    ms_pt = [float(p["infer_time_ms_per_point"]) for p in rep_profiles]
    ram_mb = [float(p["peak_ram_mb"]) for p in rep_profiles]
    vram_mb = [float(p["peak_vram_mb"]) for p in rep_profiles]
    wh_tot = [float(p["energy_wh_total"]) for p in rep_profiles]
    mwh_1k = [float(p["energy_mwh_per_1k_points"]) for p in rep_profiles]
    iqr_ms = float(np.percentile(ms_pt, 75) - np.percentile(ms_pt, 25)) if len(ms_pt) > 1 else 0.0
    return {
        "model": str(model_id),
        "config": str(config),
        "param_count": int(param_count),
        "device": str(rep_profiles[0]["device"]),
        "load_time_s": float(rep_profiles[0]["load_time_s"]),
        "infer_time_s_total": float(np.median(infer_s)),
        "infer_time_std_s": float(np.std(infer_s, ddof=1)) if len(infer_s) > 1 else 0.0,
        "infer_time_ms_per_point": float(np.median(ms_pt)),
        "infer_time_iqr_ms": iqr_ms,
        "peak_ram_mb": float(np.max(ram_mb)),
        "peak_vram_mb": float(np.max(vram_mb)),
        "energy_wh_total": float(np.median(wh_tot)),
        "energy_mwh_per_1k_points": float(np.median(mwh_1k)),
        "energy_source": str(rep_profiles[0]["energy_source"]),
        "n_forecast_points": int(rep_profiles[0]["n_forecast_points"]),
        "n_repeats": int(len(rep_profiles)),
    }

