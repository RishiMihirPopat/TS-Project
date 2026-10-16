"""Phase 3: Per-series and per-configuration time-series meta-feature computation.

Implements the exact definitions from Section 4 (Phase 3) of the Build Plan:
- `trend_strength`, `seasonal_strength` via STL decomposition (Hyndman & Athanasopoulos FPP3 §4.3):
    trend_strength = max(0, 1 - Var(R) / Var(T + R))
    seasonal_strength = max(0, 1 - Var(R) / Var(S + R))
- `spectral_entropy` (normalized Welch PSD Shannon entropy in [0, 1]), `forecastability = 1 - spectral_entropy`
- `hurst` exponent via Rescaled Range (R/S) analysis over log-spaced lags
- `stability` (variance of non-overlapping window means on z-scored series)
- `lumpiness` (variance of non-overlapping window variances on z-scored series)
- `length`, `log_length`, `frequency_group`, `n_series`
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import antropy as ant
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

META_FEATURE_NAMES: List[str] = [
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "forecastability",
    "hurst",
    "stability",
    "lumpiness",
    "length",
    "log_length",
]


def _stl_trend_and_seasonal_strength(x: np.ndarray, period: int) -> Dict[str, float]:
    """Compute Hyndman STL trend_strength and seasonal_strength."""
    n = len(x)
    eff_period = int(period) if (period >= 2 and n >= 2 * period) else 2
    if n < 2 * eff_period or np.std(x) < 1e-12:
        return {"trend_strength": 0.0, "seasonal_strength": 0.0}

    # Ensure odd seasonal window >= 7 for statsmodels STL
    seasonal_win = max(7, eff_period if eff_period % 2 == 1 else eff_period + 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = STL(x, period=eff_period, seasonal=seasonal_win, robust=True).fit()
        rem = np.asarray(res.resid, dtype=float)
        tr = np.asarray(res.trend, dtype=float)
        seas = np.asarray(res.seasonal, dtype=float)

        var_rem = float(np.var(rem, ddof=1))
        var_deseas = float(np.var(tr + rem, ddof=1))
        var_detrend = float(np.var(seas + rem, ddof=1))

        trend_str = max(0.0, min(1.0, 1.0 - var_rem / max(var_deseas, 1e-12)))
        seas_str = (
            max(0.0, min(1.0, 1.0 - var_rem / max(var_detrend, 1e-12)))
            if period >= 2 and n >= 2 * period
            else 0.0
        )
        return {"trend_strength": float(trend_str), "seasonal_strength": float(seas_str)}
    except Exception:
        return {"trend_strength": 0.0, "seasonal_strength": 0.0}


def _normalized_spectral_entropy(x: np.ndarray) -> float:
    """Compute normalized spectral entropy in [0, 1] using Welch PSD / periodogram."""
    n = len(x)
    if n < 8 or np.std(x) < 1e-12:
        return 0.0
    try:
        nperseg = min(n, max(16, n // 2))
        se = float(
            ant.spectral_entropy(x, sf=1.0, method="welch", nperseg=nperseg, normalize=True)
        )
        if np.isfinite(se):
            return float(np.clip(se, 0.0, 1.0))
    except Exception:
        pass

    # Fallback FFT periodogram Shannon entropy
    centered = x - np.mean(x)
    psd = np.abs(np.fft.rfft(centered)[1:]) ** 2
    psd_sum = float(np.sum(psd))
    if psd_sum <= 1e-12 or len(psd) <= 1:
        return 0.0
    p = psd / psd_sum
    p = p[p > 0]
    ent = -float(np.sum(p * np.log2(p))) / float(np.log2(len(psd)))
    return float(np.clip(ent, 0.0, 1.0))


def _hurst_rs(x: np.ndarray) -> float:
    """Compute Hurst exponent via Rescaled Range (R/S) analysis over sub-series lags."""
    n = len(x)
    if n < 20 or np.std(x) < 1e-12:
        return 0.5

    max_k = min(n // 2, 128)
    min_k = 8
    if max_k <= min_k:
        return 0.5

    lags = np.unique(np.geomspace(min_k, max_k, num=10).astype(int))
    lags = lags[lags >= 4]
    rs_vals = []
    valid_lags = []

    # R/S on first differences (increments) or demeaned levels:
    # Standard R/S for time-series persistence (tsfeatures / Hurst 1951) operates on demeaned chunks
    for k in lags:
        n_chunks = n // k
        if n_chunks < 1:
            continue
        chunk_rs = []
        for c in range(n_chunks):
            seg = x[c * k : (c + 1) * k]
            s_std = float(np.std(seg, ddof=1))
            if s_std <= 1e-12:
                continue
            dev = np.cumsum(seg - np.mean(seg))
            r_range = float(np.max(dev) - np.min(dev))
            if r_range > 0:
                chunk_rs.append(r_range / s_std)
        if chunk_rs:
            rs_vals.append(np.mean(chunk_rs))
            valid_lags.append(k)

    if len(valid_lags) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(valid_lags), np.log(rs_vals), 1)
    return float(np.clip(slope, 0.05, 0.99))


def _stability_and_lumpiness(x: np.ndarray, window_size: int) -> Dict[str, float]:
    """Compute tsfeatures stability and lumpiness on z-scored series over non-overlapping windows."""
    n = len(x)
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n < 6 or std <= 1e-12:
        return {"stability": 0.0, "lumpiness": 0.0}

    z = (x - float(np.mean(x))) / std
    w = max(2, min(int(window_size), max(2, n // 4)))
    n_tiles = n // w
    if n_tiles < 2:
        w = max(2, n // 3)
        n_tiles = n // w
    if n_tiles < 2:
        return {"stability": 0.0, "lumpiness": 0.0}

    trimmed = z[: n_tiles * w].reshape(n_tiles, w)
    tile_means = np.mean(trimmed, axis=1)
    tile_vars = np.var(trimmed, axis=1, ddof=1)
    return {
        "stability": float(np.var(tile_means, ddof=1)),
        "lumpiness": float(np.var(tile_vars, ddof=1)),
    }


def compute_series_meta_features(
    y: np.ndarray,
    stl_period: int = 12,
) -> Dict[str, float]:
    """Compute all Phase 3 meta-features for a single 1D time series."""
    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 4:
        return {
            "trend_strength": 0.0,
            "seasonal_strength": 0.0,
            "spectral_entropy": 1.0,
            "forecastability": 0.0,
            "hurst": 0.5,
            "stability": 0.0,
            "lumpiness": 0.0,
            "length": float(n),
            "log_length": float(np.log(max(n, 1))),
        }

    # Cap series length at 2048 most recent observations for STL/PSD speed on ultra-long series
    arr_eval = arr[-2048:] if n > 2048 else arr

    stl_feats = _stl_trend_and_seasonal_strength(arr_eval, period=stl_period)
    spec_ent = _normalized_spectral_entropy(arr_eval)
    fcast = 1.0 - spec_ent
    hurst_val = _hurst_rs(arr_eval)
    tile_width = max(4, stl_period if stl_period >= 4 else 10)
    stab_lump = _stability_and_lumpiness(arr_eval, window_size=tile_width)

    return {
        "trend_strength": float(stl_feats["trend_strength"]),
        "seasonal_strength": float(stl_feats["seasonal_strength"]),
        "spectral_entropy": float(spec_ent),
        "forecastability": float(fcast),
        "hurst": float(hurst_val),
        "stability": float(stab_lump["stability"]),
        "lumpiness": float(stab_lump["lumpiness"]),
        "length": float(n),
        "log_length": float(np.log(n)),
    }


def compute_config_meta_features(
    series_list: List[Dict[str, Any]],
    config: str,
    base_dataset: str,
    domain: str,
    freq: str,
    frequency_group: str,
    stl_period: int,
    max_series: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute per-series meta-features and aggregate to configuration level by median."""
    n_total_series = len(series_list)
    if n_total_series > max_series:
        rng = np.random.default_rng(seed)
        idxs = np.sort(rng.choice(n_total_series, size=max_series, replace=False))
        sampled = [series_list[i] for i in idxs]
    else:
        sampled = series_list

    per_series_rows = [
        compute_series_meta_features(s["target"], stl_period=stl_period)
        for s in sampled
    ]
    df_s = pd.DataFrame(per_series_rows)
    medians = df_s[META_FEATURE_NAMES].median(axis=0).to_dict()

    return {
        "config": config,
        "base_dataset": base_dataset,
        "domain": domain,
        "freq": freq,
        "frequency_group": frequency_group,
        "stl_period": int(stl_period),
        "n_series": int(n_total_series),
        "n_series_sampled": int(len(sampled)),
        **{k: float(v) for k, v in medians.items()},
    }
