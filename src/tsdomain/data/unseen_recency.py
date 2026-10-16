"""Phase 1c (Dataset 2): Post-Cutoff Recency Stratum (2025-2026 Open-Meteo Daily Series).

Fetches daily meteorological observations strictly from the post-pretraining cutoff era
(2025-01-01 through 2026-09-30; 638 daily points per series) across 20 geographically diverse
locations and 3 variables (temperature_2m_mean, shortwave_radiation_sum, wind_speed_10m_max)
= 60 series. Isolates temporal novelty (post-cutoff) from domain novelty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 20 diverse global & Indian stations -> 20 x 3 variables = 60 post-cutoff daily series
STATIONS: List[Tuple[str, float, float]] = [
    ("New_Delhi", 28.61, 77.21),
    ("Mumbai", 19.07, 72.87),
    ("Bengaluru", 12.97, 77.59),
    ("Chennai", 13.08, 80.27),
    ("Kolkata", 22.57, 88.36),
    ("Hyderabad", 17.38, 78.48),
    ("Ahmedabad", 23.02, 72.57),
    ("Pune", 18.52, 73.85),
    ("Jaipur", 26.91, 75.78),
    ("Lucknow", 26.84, 80.94),
    ("London", 51.50, -0.12),
    ("New_York", 40.71, -74.00),
    ("Tokyo", 35.68, 139.69),
    ("Sydney", -33.86, 151.20),
    ("Sao_Paulo", -23.55, -46.63),
    ("Cairo", 30.04, 31.23),
    ("Nairobi", -1.29, 36.82),
    ("Berlin", 52.52, 13.40),
    ("Toronto", 43.65, -79.38),
    ("Singapore", 1.35, 103.81),
]

DAILY_VARS: List[str] = [
    "temperature_2m_mean",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
]


def build_unseen_recency_dataset(
    raw_dir: Path = Path("data/raw"),
    out_parquet: Path = Path("data/processed/unseen_recency.parquet"),
    start_date: str = "2025-01-01",
    end_date: str = "2026-09-30",
) -> pd.DataFrame:
    """Fetch and regularize 60 post-cutoff (2025-2026) daily series from Open-Meteo."""
    if out_parquet.exists():
        df_cached = pd.read_parquet(out_parquet)
        if len(df_cached["item_id"].unique()) >= 50:
            return df_cached

    raw_cache = raw_dir / "open_meteo_recency_2025_2026.json"
    raw_cache.parent.mkdir(parents=True, exist_ok=True)

    if raw_cache.exists():
        payload = json.loads(raw_cache.read_text(encoding="utf-8"))
    else:
        lats = ",".join(f"{lat:.2f}" for _, lat, _ in STATIONS)
        lons = ",".join(f"{lon:.2f}" for _, _, lon in STATIONS)
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(DAILY_VARS),
            "timezone": "UTC",
        }
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        raw_cache.write_text(json.dumps(payload), encoding="utf-8")

    if isinstance(payload, dict):
        payload = [payload]

    rows: List[pd.DataFrame] = []
    for (city_name, _, _), loc_data in zip(STATIONS, payload):
        daily = loc_data["daily"]
        timestamps = pd.to_datetime(daily["time"])
        for var_name in DAILY_VARS:
            vals = pd.Series(daily[var_name], dtype=float).interpolate(method="linear").ffill().bfill()
            item_id = f"{city_name}__{var_name}"
            rows.append(
                pd.DataFrame(
                    {
                        "item_id": item_id,
                        "timestamp": timestamps,
                        "target": vals.to_numpy(dtype=float),
                    }
                )
            )

    final_df = pd.concat(rows, ignore_index=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_parquet, index=False)
    return final_df
