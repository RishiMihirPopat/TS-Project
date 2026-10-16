# Architecture & Engineering Decisions Log (`docs/decisions.md`)

All non-obvious engineering, methodological, and external-fact decisions are recorded here chronologically in accordance with Working Rules 2, 3, and 5.

## 2026-10-16 — Phase 0: Environment, Build Tooling, and Baseline Reference

1. **Python 3.11 via `uv` and Cross-Platform Make Runner:**
   - *Context:* The local machine runs Windows with `uv` (`C:\Users\rishi\.local\bin\uv.exe`) and `cpython-3.11.17-windows-x86_64` installed, while default `python.exe` is Python 3.14 and GNU `make` / `git.exe` are not on the system `PATH`.
   - *Decision:* Provision `.venv` explicitly with Python 3.11 (`uv venv .venv --python 3.11`), provide both a POSIX `Makefile` and a PowerShell `make.ps1` runner (`.\make.ps1 <target>`), and include `dulwich` in `pyproject.toml` so standard Git commits are written directly to `.git/` after each phase/task.
2. **Hardware Detection (`--device {auto,cpu,cuda}`):**
   - *Context:* `nvidia-smi` is not present on this machine (CPU-only host).
   - *Decision:* Every CLI and model runner implements `--device {auto,cpu,cuda}` where `auto` inspects `torch.cuda.is_available()` and falls back cleanly to `cpu`. Energy profiling via `codecarbon` records `energy_method` (`RAPL` / `TDP_estimate` on CPU; `NVML` when CUDA is available).
3. **`statsmodels` vs `statsforecast` ETS/Theta/ARIMA Parity (Phase 0):**
   - *Context:* The Milestone 1 baseline table on `AirPassengers` (`ETS MASE = 1.278`, `Theta = 2.213`, `SARIMA(1,1,1)(1,1,1)12 = 2.446`, `Seasonal Naive = 2.494`, `Drift = 3.206`) was generated with `statsmodels` (`ExponentialSmoothing(trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12)`, `ThetaModel(period=12)`, `SARIMAX((1,1,1),(1,1,1,12))`). `statsforecast`'s `AutoETS` searches across all $(E,T,S)$ states by AICc rather than fixing `(A_d, M)`.
   - *Decision:* Keep the exact `statsmodels` implementation in `seed/baseline_model.py` and `src/tsdomain/baselines/classical.py` (tested at strict tolerance $\pm 0.02$), alongside the `statsforecast` suite (`SeasonalNaive`, `Naive`, `RandomWalkWithDrift`, `AutoETS`, `AutoTheta`, `AutoARIMA`) used for fast multi-series benchmarking across GIFT-Eval and unseen datasets.

## 2026-10-16 — Phase 1: Data Acquisition, Leaderboard Ingestion, and Registry

4. **GIFT-Eval Leaderboard & Windowing Exactness (`src/tsdomain/data/leaderboard.py`, `src/tsdomain/data/gift_eval.py`):**
   - *Verified Fact:* In `SalesforceAIResearch/gift-eval`, each model subfolder under `results/<MODEL>/` contains `all_results.csv` (97 configurations, 15 columns) and `config.json` (`testdata_leakage`, `model_type`). In `src/gift_eval/data.py`, windowing uses `TEST_SPLIT = 0.1`, `MAX_WINDOW = 20` (`windows = 1` for `m4_*`), and `gluonts.time_feature.get_seasonality(freq)` maps `D -> 1` and `W -> 1` (while `M -> 12`, `Q -> 4`, `H -> 24`, `15T -> 96`, `10T -> 144`, `5T -> 288`, `10S -> 360`).
   - *Decision:* Replicate GluonTS `get_seasonality(freq)` for MASE scaling (`season_length`) to ensure exact parity with published GIFT-Eval `Seasonal_Naive` MASE, while separately providing natural STL periods (`stl_period = 7` for `D`, `52` for `W`) in `get_stl_period()` so STL seasonal strength in Phase 3 captures weekly/annual cycles on daily and weekly series.
5. **Unseen Dataset 1 — Indian Wholesale Agricultural Mandi Prices (`unseen_mandi`):**
   - *Verified Fact:* Direct `agmarknet.gov.in` bulk downloads require interactive ASP.NET form state, and `Viharikvs/TFSMAG` on Hugging Face is manually gated (`"gated": "manual"`). However, the public GitHub repository `iancovert/Agmarknet` hosts 4M+ daily wholesale mandi records across 12 commodities (2008–2017) from `agmarknet.gov.in`.
   - *Decision:* Ingest 5 continuous years (`2012–2016`, 260 weeks) across 6 major crops (`Wheat`, `Maize`, `Soyabean`, `Lentil (Masur)`, `Green Gram (Moong)`, `Arhar (Tur-Red Gram)`) and select the top 60 market×commodity series with highest weekly coverage, regularized onto a `W-MON` grid (`data/processed/unseen_mandi.parquet`).
6. **Unseen Dataset 2 — Post-Cutoff 2025–2026 Recency Stratum (`unseen_recency`):**
   - *Decision:* Query Open-Meteo ERA5 Historical Archive (`https://archive-api.open-meteo.com/v1/archive`) from `2025-01-01` to `2026-09-30` (638 daily observations per series) across 20 global/Indian cities and 3 meteorological variables (`temperature_2m_mean`, `shortwave_radiation_sum`, `wind_speed_10m_max`) = 60 daily series (`data/processed/unseen_recency.parquet`), isolating post-cutoff temporal novelty from domain novelty.
7. **S&P 500 Daily OHLC Fallback (`src/tsdomain/data/finance.py`):**
   - *Verified Fact:* `https://stooq.com/q/d/l/?s=^spx&i=d` returned an HTML redirect page at build time.
   - *Decision:* `fetch_spx_daily_ohlc()` attempts Stooq first and automatically falls back to the Yahoo Finance v8 chart endpoint (`%5EGSPC`, 1990–2026, 9,247 trading days), caching `data/raw/spx_daily.csv` and `data/processed/spx_daily.parquet`.
