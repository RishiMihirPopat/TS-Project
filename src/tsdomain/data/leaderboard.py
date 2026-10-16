"""Phase 1b: Ingest published GIFT-Eval leaderboard results from SalesforceAIResearch/gift-eval."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/main"

# Canonical model type categorization for downstream analysis
MODEL_TYPE_MAP: Dict[str, str] = {
    "Naive": "statistical",
    "Seasonal_Naive": "statistical",
    "Auto_Arima": "statistical",
    "Auto_ETS": "statistical",
    "Auto_Theta": "statistical",
    "PatchTST": "deep-learning",
    "N-BEATS": "deep-learning",
    "DLinear": "deep-learning",
    "TFT": "deep-learning",
    "DeepAR": "deep-learning",
    "Chronos_small": "foundation",
    "Chronos_base": "foundation",
    "Chronos_large": "foundation",
    "chronos_bolt_small": "foundation",
    "chronos_bolt_base": "foundation",
    "Chronos-2": "foundation",
    "Moirai_small": "foundation",
    "Moirai_base": "foundation",
    "Moirai_large": "foundation",
    "Moirai2": "foundation",
    "TimesFM": "foundation",
    "timesfm_2_0_500m": "foundation",
    "TimesFM-2.5": "foundation",
    "TTM-R1-Pretrained": "foundation",
    "TTM-R2-Pretrained": "foundation",
}


def fetch_dataset_properties(raw_cache_dir: Path = Path("data/raw/gift_eval_meta")) -> Dict[str, Dict]:
    """Download and cache GIFT-Eval's official dataset_properties.json."""
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_cache_dir / "dataset_properties.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = f"{GITHUB_RAW_BASE}/notebooks/dataset_properties.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.json()


def fetch_single_model_leaderboard(
    model_folder: str,
    raw_cache_dir: Path = Path("data/raw/leaderboard"),
) -> pd.DataFrame:
    """Download config.json and all_results.csv for a single GIFT-Eval model folder."""
    model_cache = raw_cache_dir / model_folder
    model_cache.mkdir(parents=True, exist_ok=True)
    csv_cache = model_cache / "all_results.csv"
    cfg_cache = model_cache / "config.json"

    if not cfg_cache.exists():
        cfg_url = f"{GITHUB_RAW_BASE}/results/{model_folder}/config.json"
        r_cfg = requests.get(cfg_url, timeout=30)
        if r_cfg.status_code == 200:
            cfg_cache.write_text(r_cfg.text, encoding="utf-8")
        else:
            cfg_cache.write_text(json.dumps({"model": model_folder}), encoding="utf-8")

    if not csv_cache.exists():
        csv_url = f"{GITHUB_RAW_BASE}/results/{model_folder}/all_results.csv"
        r_csv = requests.get(csv_url, timeout=30)
        r_csv.raise_for_status()
        csv_cache.write_text(r_csv.text, encoding="utf-8")

    cfg_meta = json.loads(cfg_cache.read_text(encoding="utf-8"))
    df_raw = pd.read_csv(csv_cache)

    # Parse dataset config string: e.g. "solar/H/short" or "m4_monthly/M/short"
    parts = df_raw["dataset"].astype(str).str.split("/", expand=True)
    base_dataset = parts[0].str.strip()
    freq = parts[1].str.strip() if parts.shape[1] > 1 else "unknown"
    term = parts[2].str.strip() if parts.shape[1] > 2 else "short"

    leakage_str = str(cfg_meta.get("testdata_leakage", "No")).strip().lower()
    leakage_flag = leakage_str in {"yes", "true", "1"}

    raw_model_type = str(cfg_meta.get("model_type", MODEL_TYPE_MAP.get(model_folder, "foundation"))).lower()
    canonical_type = MODEL_TYPE_MAP.get(model_folder, "foundation" if "pretrained" in raw_model_type or "zero-shot" in raw_model_type else raw_model_type)

    out_df = pd.DataFrame(
        {
            "model": model_folder,
            "raw_model_name": df_raw["model"].astype(str),
            "model_type": canonical_type,
            "raw_model_type": raw_model_type,
            "config": df_raw["dataset"].astype(str).str.strip(),
            "base_dataset": base_dataset,
            "freq": freq,
            "term": term,
            "domain": df_raw["domain"].astype(str).str.strip(),
            "num_variates": pd.to_numeric(df_raw["num_variates"], errors="coerce").fillna(1).astype(int),
            "mase": pd.to_numeric(df_raw["eval_metrics/MASE[0.5]"], errors="coerce"),
            "crps": pd.to_numeric(df_raw["eval_metrics/mean_weighted_sum_quantile_loss"], errors="coerce"),
            "mse_mean": pd.to_numeric(df_raw.get("eval_metrics/MSE[mean]"), errors="coerce"),
            "mae_median": pd.to_numeric(df_raw.get("eval_metrics/MAE[0.5]"), errors="coerce"),
            "smape": pd.to_numeric(df_raw.get("eval_metrics/sMAPE[0.5]"), errors="coerce"),
            "msis": pd.to_numeric(df_raw.get("eval_metrics/MSIS"), errors="coerce"),
            "rmse_mean": pd.to_numeric(df_raw.get("eval_metrics/RMSE[mean]"), errors="coerce"),
            "nrmse_mean": pd.to_numeric(df_raw.get("eval_metrics/NRMSE[mean]"), errors="coerce"),
            "nd": pd.to_numeric(df_raw.get("eval_metrics/ND[0.5]"), errors="coerce"),
            "test_data_leakage_flag": bool(leakage_flag),
            "org": str(cfg_meta.get("org", "")),
            "model_link": str(cfg_meta.get("model_link", "")),
        }
    )
    # Deduplicate on (model, config) if any submission CSV had duplicate rows
    out_df = out_df.drop_duplicates(subset=["model", "config"], keep="first").reset_index(drop=True)
    return out_df


def ingest_gift_eval_leaderboard(
    model_folders: Optional[List[str]] = None,
    out_parquet: Path = Path("results/metrics/leaderboard.parquet"),
) -> pd.DataFrame:
    """Download and consolidate GIFT-Eval leaderboard results into results/metrics/leaderboard.parquet."""
    if model_folders is None:
        model_folders = list(MODEL_TYPE_MAP.keys())

    frames: List[pd.DataFrame] = []
    for folder in model_folders:
        df_m = fetch_single_model_leaderboard(folder)
        frames.append(df_m)

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["model", "config"], keep="first").reset_index(drop=True)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_parquet, index=False)
    return full_df
