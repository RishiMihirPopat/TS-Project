"""Phase 1d / Phase 7: S&P 500 Daily OHLC Loader (1990 onward) and Garman-Klass Volatility.

Attempts Stooq first, falls back to Yahoo Finance v8 chart API (%5EGSPC), or reads an
owner-supplied `data/raw/spx_daily.csv` with columns `Date,Open,High,Low,Close`.
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def fetch_spx_daily_ohlc(
    raw_csv_path: Path = Path("data/raw/spx_daily.csv"),
    processed_parquet_path: Path = Path("data/processed/spx_daily.parquet"),
    start_year: int = 1990,
) -> pd.DataFrame:
    """Acquire S&P 500 daily OHLC from 1990 onward and compute Garman-Klass volatility."""
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    processed_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df_ohlc: pd.DataFrame | None = None

    if raw_csv_path.exists() and raw_csv_path.stat().st_size > 10_000:
        df_ohlc = pd.read_csv(raw_csv_path)
    else:
        # 1. Try Stooq CSV endpoint
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r_stooq = requests.get("https://stooq.com/q/d/l/?s=^spx&i=d", headers=headers, timeout=20)
            if r_stooq.status_code == 200 and "Date,Open,High,Low,Close" in r_stooq.text[:100]:
                df_ohlc = pd.read_csv(io.StringIO(r_stooq.text))
        except Exception:
            df_ohlc = None

        # 2. Fallback to Yahoo Finance v8 chart API (%5EGSPC)
        if df_ohlc is None or df_ohlc.empty:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            p1 = int(dt.datetime(start_year, 1, 1, tzinfo=dt.timezone.utc).timestamp())
            p2 = int(dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={p1}&period2={p2}&interval=1d"
            r_yf = requests.get(url, headers=headers, timeout=30)
            r_yf.raise_for_status()
            res = r_yf.json()["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            dates = [
                dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).strftime("%Y-%m-%d")
                for t in ts
            ]
            df_ohlc = pd.DataFrame(
                {
                    "Date": dates,
                    "Open": q["open"],
                    "High": q["high"],
                    "Low": q["low"],
                    "Close": q["close"],
                }
            )
            df_ohlc = df_ohlc.dropna().reset_index(drop=True)
            df_ohlc.to_csv(raw_csv_path, index=False)

    df_ohlc["Date"] = pd.to_datetime(df_ohlc["Date"])
    df_ohlc = (
        df_ohlc[df_ohlc["Date"] >= pd.Timestamp(f"{start_year}-01-01")]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    for col in ["Open", "High", "Low", "Close"]:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors="coerce")
    df_ohlc = df_ohlc.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    # Compute Garman-Klass (1980) daily variance estimator:
    # sigma^2_GK = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    hl = np.log(df_ohlc["High"] / df_ohlc["Low"])
    co = np.log(df_ohlc["Close"] / df_ohlc["Open"])
    gk_var = 0.5 * (hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (co ** 2)
    # Floor at 1e-8 to avoid log(0) on flat days
    gk_var = np.maximum(gk_var, 1e-8)

    df_ohlc["gk_var"] = gk_var
    df_ohlc["gk_vol"] = np.sqrt(gk_var)
    df_ohlc["log_gk_var"] = np.log(gk_var)
    df_ohlc["log_return"] = np.log(df_ohlc["Close"] / df_ohlc["Close"].shift(1)).fillna(0.0)

    df_ohlc.to_parquet(processed_parquet_path, index=False)
    return df_ohlc
