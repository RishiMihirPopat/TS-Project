"""Phase 1a: GIFT-Eval Dataset Loader, Configuration Catalog, and Rolling Window Generator.

Replicates the exact dataset mapping, prediction lengths, seasonality, and rolling test
window splits from `SalesforceAIResearch/gift-eval` (`src/gift_eval/data.py`), and provides
a unified window generator for both GIFT-Eval configurations and the unseen-domain datasets.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP: Dict[str, int] = {
    "A": 6,
    "Y": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP: Dict[str, int] = {
    "A": 6,
    "Y": 6,
    "Q": 8,
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

TERM_MULTIPLIERS: Dict[str, int] = {
    "short": 1,
    "medium": 10,
    "long": 15,
}

# GluonTS get_seasonality(freq) exact mapping used by GIFT-Eval
GLUONTS_SEASONALITY_MAP: Dict[str, int] = {
    "A": 1,
    "Y": 1,
    "Q": 4,
    "M": 12,
    "W": 1,
    "D": 1,
    "B": 5,
    "H": 24,
    "15T": 96,
    "10T": 144,
    "5T": 288,
    "T": 1440,
    "10S": 360,
    "S": 3600,
}

# STL period for meta-feature computation (where GluonTS uses m=1 for D/W, STL uses natural cycle 7/52)
STL_PERIOD_MAP: Dict[str, int] = {
    "A": 1,
    "Y": 1,
    "Q": 4,
    "M": 12,
    "W": 52,
    "D": 7,
    "B": 5,
    "H": 24,
    "15T": 96,
    "10T": 144,
    "5T": 288,
    "T": 144,
    "10S": 360,
    "S": 60,
}

# Mapping from leaderboard base_dataset + freq to Hugging Face Salesforce/GiftEval subdirectory
BASE_TO_HF_PREFIX: Dict[str, str] = {
    "loop_seattle": "LOOP_SEATTLE",
    "m_dense": "M_DENSE",
    "sz_taxi": "SZ_TAXI",
    "saugeen": "saugeenday",
    "temperature_rain": "temperature_rain_with_missing",
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
}

SINGLE_DIR_DATASETS = {
    "bizitobs_application",
    "bizitobs_service",
    "car_parts_with_missing",
    "covid_deaths",
    "hospital",
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_weekly",
    "m4_yearly",
    "restaurant",
    "temperature_rain_with_missing",
}


def config_to_hf_subdir(config: str) -> str:
    """Map a GIFT-Eval leaderboard config (e.g. 'loop_seattle/D/short') to its HF repo subdir."""
    parts = config.strip().split("/")
    base = parts[0].strip()
    freq = parts[1].strip() if len(parts) > 1 else ""
    hf_base = BASE_TO_HF_PREFIX.get(base.lower(), base)
    if hf_base in SINGLE_DIR_DATASETS:
        return hf_base
    return f"{hf_base}/{freq}"


def parse_base_freq_code(freq_str: str) -> str:
    """Normalize pandas/GluonTS frequency string to base unit (A, Q, M, W, D, H, T, S) and multiplier."""
    f = str(freq_str).strip()
    upper = f.upper()
    if upper.startswith(("15T", "15MIN")):
        return "15T"
    if upper.startswith(("10T", "10MIN")):
        return "10T"
    if upper.startswith(("5T", "5MIN")):
        return "5T"
    if upper.startswith(("10S", "10SEC")):
        return "10S"
    if upper.startswith(("A", "Y")):
        return "A"
    if upper.startswith("Q"):
        return "Q"
    if upper.startswith("M") and not upper.startswith("MIN"):
        return "M"
    if upper.startswith("W"):
        return "W"
    if upper.startswith("D") or upper.startswith("B"):
        return "D"
    if upper.startswith("H"):
        return "H"
    if "T" in upper or "MIN" in upper:
        return "T"
    if "S" in upper:
        return "S"
    return "D"


def get_gift_eval_prediction_length(config: str, freq_str: Optional[str] = None) -> int:
    """Return the exact prediction length for a GIFT-Eval config."""
    parts = config.strip().split("/")
    base = parts[0].lower()
    freq = freq_str or (parts[1] if len(parts) > 1 else "D")
    term = parts[2].lower() if len(parts) > 2 else "short"

    norm_f = parse_base_freq_code(freq)
    unit = norm_f[-1]  # last char is A, Q, M, W, D, H, T, S
    base_len = M4_PRED_LENGTH_MAP[unit] if "m4" in base else PRED_LENGTH_MAP[unit]
    return int(TERM_MULTIPLIERS.get(term, 1) * base_len)


def get_gift_eval_season_length(config: str, freq_str: Optional[str] = None) -> int:
    """Return the exact seasonality parameter m used by GIFT-Eval for MASE scaling (W->1, D->1)."""
    parts = config.strip().split("/")
    freq = freq_str or (parts[1] if len(parts) > 1 else "D")
    norm_f = parse_base_freq_code(freq)
    return int(GLUONTS_SEASONALITY_MAP.get(norm_f, GLUONTS_SEASONALITY_MAP.get(norm_f[-1], 1)))


def get_stl_period(config: str, freq_str: Optional[str] = None) -> int:
    """Return the seasonal period for STL trend/seasonality decomposition."""
    if config.startswith("unseen_mandi"):
        return 52
    if config.startswith("unseen_recency"):
        return 7
    parts = config.strip().split("/")
    freq = freq_str or (parts[1] if len(parts) > 1 else "D")
    norm_f = parse_base_freq_code(freq)
    return int(STL_PERIOD_MAP.get(norm_f, STL_PERIOD_MAP.get(norm_f[-1], 7)))


def get_frequency_group(freq_str: str) -> str:
    """Categorize frequency into coarse groups for mixed-effects modeling."""
    norm_f = parse_base_freq_code(freq_str)
    if norm_f in {"A", "Y", "Q"}:
        return "low_freq_AQ"
    if norm_f == "M":
        return "monthly"
    if norm_f == "W":
        return "weekly"
    if norm_f == "D":
        return "daily"
    if norm_f == "H":
        return "hourly"
    return "sub_hourly"


def _pandas_freq_compatible(freq_str: str) -> str:
    """Convert GluonTS/legacy pandas freq strings to pandas 2.2/3.0 compatible offset aliases."""
    f = str(freq_str).strip()
    remap = {
        "M": "ME",
        "Q": "QE",
        "A": "YE",
        "Y": "YE",
        "A-DEC": "YE-DEC",
        "Q-DEC": "QE-DEC",
        "H": "h",
        "15T": "15min",
        "10T": "10min",
        "5T": "5min",
        "T": "min",
        "10S": "10s",
        "S": "s",
    }
    return remap.get(f, f)


def download_and_load_gift_eval_series(
    config: str,
    raw_cache_dir: Path = Path("data/raw/gift_eval"),
    max_raw_series: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Download the Arrow shard(s) for a GIFT-Eval config and return univariate series dicts + min_len.

    Also returns `_min_series_length` computed across the entire HF dataset (matching
    `Dataset._min_series_length` in `gift_eval/data.py`) so window count matches GIFT-Eval.
    """
    hf_subdir = config_to_hf_subdir(config)
    local_dir = raw_cache_dir / hf_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    arrow_local = local_dir / "data-00000-of-00001.arrow"

    if not arrow_local.exists():
        downloaded = hf_hub_download(
            repo_id="Salesforce/GiftEval",
            filename=f"{hf_subdir}/data-00000-of-00001.arrow",
            repo_type="dataset",
            local_dir=str(raw_cache_dir),
        )
        arrow_local = Path(downloaded)

    ds = datasets.Dataset.from_file(str(arrow_local))
    total_rows = len(ds)

    # Compute exact _min_series_length across all rows in the shard
    sample_target = ds[0]["target"]
    is_multivariate = len(sample_target) > 0 and isinstance(sample_target[0], (list, np.ndarray))

    all_lengths: List[int] = []
    # If dataset has thousands of series (like m4_monthly with 48,000), inspect arrow column lengths directly
    if not is_multivariate:
        import pyarrow.compute as pc

        lengths_arr = pc.list_value_length(ds.data.column("target")).to_numpy()
        min_series_len = int(np.min(lengths_arr))
    else:
        import pyarrow.compute as pc

        lengths_arr = pc.list_value_length(
            pc.list_flatten(pc.list_slice(ds.data.column("target"), 0, 1))
        ).to_numpy()
        min_series_len = int(np.min(lengths_arr))

    # Select indices deterministically if max_raw_series is set
    if max_raw_series is not None and total_rows > max_raw_series:
        rng = np.random.default_rng(42)
        selected_indices = np.sort(rng.choice(total_rows, size=max_raw_series, replace=False)).tolist()
    else:
        selected_indices = list(range(total_rows))

    univariate_series: List[Dict[str, Any]] = []
    for idx in selected_indices:
        row = ds[int(idx)]
        item_id = str(row["item_id"])
        start_dt = pd.Timestamp(row["start"])
        freq_str = str(row["freq"])
        target = row["target"]

        if is_multivariate:
            for dim_idx, dim_vals in enumerate(target):
                arr = np.asarray(dim_vals, dtype=float)
                univariate_series.append(
                    {
                        "item_id": f"{item_id}_dim{dim_idx}",
                        "start": start_dt,
                        "freq": freq_str,
                        "target": arr,
                    }
                )
                if max_raw_series is not None and len(univariate_series) >= max_raw_series:
                    break
        else:
            arr = np.asarray(target, dtype=float)
            univariate_series.append(
                {
                    "item_id": item_id,
                    "start": start_dt,
                    "freq": freq_str,
                    "target": arr,
                }
            )
        if max_raw_series is not None and len(univariate_series) >= max_raw_series:
            break

    return univariate_series, min_series_len


def _impute_last_value(arr: np.ndarray) -> np.ndarray:
    """Replicate GluonTS LastValueImputation (forward-fill then back-fill leading NaNs with 0.0)."""
    if not np.isnan(arr).any():
        return arr
    s = pd.Series(arr, dtype=float).ffill().fillna(0.0)
    return s.to_numpy(dtype=float)


def load_series_for_config(
    config: str,
    max_series: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Load univariate series list + (prediction_length, season_length, gift_eval_windows) for any config."""
    if config.startswith("unseen_mandi"):
        from tsdomain.data.unseen_mandi import build_unseen_mandi_dataset

        df = build_unseen_mandi_dataset()
        series_list: List[Dict[str, Any]] = []
        for item_id, grp in df.groupby("item_id", sort=True):
            grp_sorted = grp.sort_values("timestamp")
            series_list.append(
                {
                    "item_id": str(item_id),
                    "start": pd.Timestamp(grp_sorted["timestamp"].iloc[0]),
                    "freq": "W-MON",
                    "target": grp_sorted["target"].to_numpy(dtype=float),
                    "timestamps": grp_sorted["timestamp"].tolist(),
                }
            )
        if max_series is not None and len(series_list) > max_series:
            rng = np.random.default_rng(seed)
            idxs = np.sort(rng.choice(len(series_list), size=max_series, replace=False))
            series_list = [series_list[i] for i in idxs]
        min_len = min(len(s["target"]) for s in series_list)
        pred_len = 8
        season_len = 52
        w = min(max(1, math.ceil(TEST_SPLIT * min_len / pred_len)), MAX_WINDOW)
        return series_list, pred_len, season_len, w

    if config.startswith("unseen_recency"):
        from tsdomain.data.unseen_recency import build_unseen_recency_dataset

        df = build_unseen_recency_dataset()
        series_list = []
        for item_id, grp in df.groupby("item_id", sort=True):
            grp_sorted = grp.sort_values("timestamp")
            series_list.append(
                {
                    "item_id": str(item_id),
                    "start": pd.Timestamp(grp_sorted["timestamp"].iloc[0]),
                    "freq": "D",
                    "target": grp_sorted["target"].to_numpy(dtype=float),
                    "timestamps": grp_sorted["timestamp"].tolist(),
                }
            )
        if max_series is not None and len(series_list) > max_series:
            rng = np.random.default_rng(seed)
            idxs = np.sort(rng.choice(len(series_list), size=max_series, replace=False))
            series_list = [series_list[i] for i in idxs]
        min_len = min(len(s["target"]) for s in series_list)
        pred_len = 30
        season_len = 7
        w = min(max(1, math.ceil(TEST_SPLIT * min_len / pred_len)), MAX_WINDOW)
        return series_list, pred_len, season_len, w

    # Standard GIFT-Eval dataset configuration
    series_list, min_series_len = download_and_load_gift_eval_series(
        config, max_raw_series=max_series
    )
    freq_str = series_list[0]["freq"] if series_list else config.split("/")[1]
    pred_len = get_gift_eval_prediction_length(config, freq_str=freq_str)
    season_len = get_gift_eval_season_length(config, freq_str=freq_str)
    if "m4" in config.lower():
        w = 1
    else:
        w = min(max(1, math.ceil(TEST_SPLIT * min_series_len / pred_len)), MAX_WINDOW)
    return series_list, pred_len, season_len, w


def generate_rolling_windows(
    config: str,
    max_series: Optional[int] = 60,
    max_windows_per_series: Optional[int] = 2,
    seed: int = 42,
    exact_gift_eval: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate rolling evaluation windows for a dataset configuration.

    If `exact_gift_eval=True`, uses all series and all `w` GIFT-Eval rolling windows
    without capping, for exact leaderboard cross-checking in Phase 2.
    """
    eff_max_series = None if exact_gift_eval else max_series
    series_list, pred_len, season_len, total_w = load_series_for_config(
        config, max_series=eff_max_series, seed=seed
    )

    if exact_gift_eval or max_windows_per_series is None or total_w <= max_windows_per_series:
        window_indices = list(range(total_w))
    else:
        # Take the most recent `max_windows_per_series` windows
        window_indices = list(range(total_w - max_windows_per_series, total_w))

    windows_out: List[Dict[str, Any]] = []
    for s in series_list:
        raw_target = np.asarray(s["target"], dtype=float)
        imputed_target = _impute_last_value(raw_target)
        n = len(imputed_target)
        pd_freq = _pandas_freq_compatible(s["freq"])

        if "timestamps" in s:
            ts_all = list(s["timestamps"])
        else:
            try:
                ts_all = list(pd.date_range(start=s["start"], periods=n, freq=pd_freq))
            except Exception:
                ts_all = list(pd.date_range(start="2000-01-01", periods=n, freq="D"))

        for w_idx in window_indices:
            train_end = n - (total_w - w_idx) * pred_len
            test_end = train_end + pred_len
            if train_end < max(4, season_len + 1) or test_end > n:
                continue
            y_train = imputed_target[:train_end]
            y_test = raw_target[train_end:test_end]
            # If raw test has NaNs, impute so metrics are well-defined or mask
            if np.isnan(y_test).any():
                y_test = imputed_target[train_end:test_end]
            test_ts = ts_all[train_end:test_end]

            windows_out.append(
                {
                    "item_id": s["item_id"],
                    "window_id": int(w_idx),
                    "y_train": y_train,
                    "y_test": y_test,
                    "test_timestamps": test_ts,
                    "prediction_length": pred_len,
                    "season_length": season_len,
                    "freq": s["freq"],
                }
            )

    info = {
        "config": config,
        "n_series": len(series_list),
        "n_windows_per_series": len(window_indices),
        "total_gift_eval_windows": total_w,
        "total_eval_windows": len(windows_out),
        "prediction_length": pred_len,
        "season_length": season_len,
    }
    return windows_out, info


def build_gift_eval_catalog(
    leaderboard_parquet: Path = Path("results/metrics/leaderboard.parquet"),
    out_parquet: Path = Path("results/metrics/gift_eval_catalog.parquet"),
) -> pd.DataFrame:
    """Build and save the catalog of all 97 GIFT-Eval configurations from the leaderboard."""
    if not leaderboard_parquet.exists():
        from tsdomain.data.leaderboard import ingest_gift_eval_leaderboard

        ingest_gift_eval_leaderboard(out_parquet=leaderboard_parquet)

    lb = pd.read_parquet(leaderboard_parquet)
    sn = lb[lb["model"] == "Seasonal_Naive"].copy()
    rows = []
    for _, r in sn.iterrows():
        cfg = str(r["config"])
        freq = str(r["freq"])
        rows.append(
            {
                "config": cfg,
                "base_dataset": str(r["base_dataset"]),
                "domain": str(r["domain"]),
                "freq": freq,
                "frequency_group": get_frequency_group(freq),
                "term": str(r["term"]),
                "num_variates": int(r["num_variates"]),
                "prediction_length": get_gift_eval_prediction_length(cfg, freq_str=freq),
                "season_length": get_gift_eval_season_length(cfg, freq_str=freq),
                "stl_period": get_stl_period(cfg, freq_str=freq),
                "hf_subdir": config_to_hf_subdir(cfg),
            }
        )
    cat_df = pd.DataFrame(rows).sort_values("config").reset_index(drop=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    cat_df.to_parquet(out_parquet, index=False)
    return cat_df
