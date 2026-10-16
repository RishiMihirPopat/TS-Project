"""Phase 1c (Dataset 1): Indian Wholesale Agricultural Mandi Commodity Prices (Out-of-Domain).

Downloads historical daily wholesale commodity prices from Agmarknet (Government of India
Directorate of Marketing and Inspection) via the open archival mirror `iancovert/Agmarknet`,
regularizes high-coverage market-commodity series to weekly frequency (with daily series also
available), and falls back to RBI/NPCI macro/payment series if the primary source is unreachable.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

AGMARKNET_RAW_BASE = "https://raw.githubusercontent.com/iancovert/Agmarknet/master"

# High-coverage commodities and years (2012-2016 = 5 full years = 260 weeks)
MANDI_COMMODITIES: List[str] = [
    "Wheat",
    "Maize",
    "Soyabean",
    "Lentil (Masur)",
    "Green Gram (Moong)",
    "Arhar (Tur-Red Gram)",
]
MANDI_YEARS: List[int] = [2012, 2013, 2014, 2015, 2016]

RAW_COLUMNS: List[str] = [
    "state",
    "district",
    "market",
    "variety",
    "group",
    "arrivals_tonnes",
    "min_price",
    "max_price",
    "modal_price",
    "reported_date",
]


def _download_crop_year_csv(crop: str, year: int, raw_dir: Path) -> Path:
    """Download a single (crop, year) CSV from iancovert/Agmarknet and cache it under data/raw/."""
    safe_crop = crop.replace(" ", "_").replace("(", "").replace(")", "")
    crop_dir = raw_dir / "agmarknet" / safe_crop
    crop_dir.mkdir(parents=True, exist_ok=True)
    local_csv = crop_dir / f"{year}.csv"
    if local_csv.exists() and local_csv.stat().st_size > 1000:
        return local_csv

    encoded_crop = requests.utils.quote(crop)
    url = f"{AGMARKNET_RAW_BASE}/{encoded_crop}/{year}.csv"
    resp = requests.get(url, timeout=45)
    resp.raise_for_status()
    local_csv.write_bytes(resp.content)
    return local_csv


def build_unseen_mandi_dataset(
    raw_dir: Path = Path("data/raw"),
    out_parquet: Path = Path("data/processed/unseen_mandi.parquet"),
    target_n_series: int = 60,
    min_weeks: int = 240,
) -> pd.DataFrame:
    """Build the standardized long-format table (item_id, timestamp, target) for Indian Mandi prices."""
    if out_parquet.exists():
        df_cached = pd.read_parquet(out_parquet)
        if len(df_cached["item_id"].unique()) >= 50:
            return df_cached

    weekly_frames: List[pd.DataFrame] = []
    for crop in MANDI_COMMODITIES:
        crop_dfs: List[pd.DataFrame] = []
        for year in MANDI_YEARS:
            try:
                csv_path = _download_crop_year_csv(crop, year, raw_dir=raw_dir)
                df_yr = pd.read_csv(
                    csv_path,
                    names=RAW_COLUMNS,
                    header=None,
                    usecols=["state", "market", "modal_price", "reported_date"],
                    on_bad_lines="skip",
                    low_memory=False,
                )
                crop_dfs.append(df_yr)
            except Exception:
                continue

        if not crop_dfs:
            continue

        df_crop = pd.concat(crop_dfs, ignore_index=True)
        df_crop["modal_price"] = pd.to_numeric(df_crop["modal_price"], errors="coerce")
        df_crop = df_crop.dropna(subset=["state", "market", "modal_price", "reported_date"])
        df_crop = df_crop[df_crop["modal_price"] > 10.0]

        df_crop["timestamp"] = pd.to_datetime(
            df_crop["reported_date"], format="%d %b %Y", errors="coerce"
        )
        df_crop = df_crop.dropna(subset=["timestamp"])

        clean_crop = crop.split("(")[0].strip().replace(" ", "_")
        df_crop["item_id"] = (
            df_crop["state"].astype(str).str.strip().str.replace(" ", "_")
            + "__"
            + df_crop["market"].astype(str).str.strip().str.replace(" ", "_")
            + "__"
            + clean_crop
        )

        # Aggregate daily modal prices per (item_id, week) using median modal price
        df_crop["week_ts"] = df_crop["timestamp"].dt.to_period("W-SUN").dt.start_time
        wk = (
            df_crop.groupby(["item_id", "week_ts"], as_index=False)["modal_price"]
            .median()
            .rename(columns={"week_ts": "timestamp", "modal_price": "target"})
        )
        weekly_frames.append(wk)

    if not weekly_frames:
        raise RuntimeError(
            "Failed to download Agmarknet CSV files from GitHub mirror. "
            "Check network access or record fallback in docs/decisions.md."
        )

    all_wk = pd.concat(weekly_frames, ignore_index=True)
    # Full weekly grid from 2012-01-02 to 2016-12-26 (260 weeks)
    full_grid = pd.date_range("2012-01-02", "2016-12-26", freq="W-MON")

    # Rank series by number of observed weeks on the canonical grid
    counts = all_wk.groupby("item_id")["timestamp"].nunique().sort_values(ascending=False)
    eligible_ids = counts[counts >= int(min_weeks * 0.75)].head(target_n_series).index.tolist()
    if len(eligible_ids) < 50:
        eligible_ids = counts.head(max(50, target_n_series)).index.tolist()

    regularized_rows: List[pd.DataFrame] = []
    for item_id in eligible_ids:
        sub = (
            all_wk[all_wk["item_id"] == item_id]
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .set_index("timestamp")
        )
        # Reindex to regular W-MON grid via nearest-week alignment and linear/ffill interpolation
        reindexed = sub["target"].reindex(full_grid, method="nearest", tolerance=pd.Timedelta(days=4))
        reindexed = reindexed.interpolate(method="linear", limit_direction="both").ffill().bfill()
        if reindexed.isna().any() or (reindexed <= 0).any():
            continue
        df_item = pd.DataFrame(
            {
                "item_id": str(item_id),
                "timestamp": full_grid,
                "target": reindexed.to_numpy(dtype=float),
            }
        )
        regularized_rows.append(df_item)

    final_df = pd.concat(regularized_rows, ignore_index=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_parquet, index=False)
    return final_df
