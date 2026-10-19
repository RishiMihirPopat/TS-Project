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

## 2026-10-16 — Phase 2: Classical Baselines and Leaderboard Cross-Check

8. **Exact GIFT-Eval `Seasonal_Naive` Leaderboard Parity (`results/metrics/leaderboard_crosscheck.parquet`):**
   - *Result:* Running `Seasonal_Naive` on the full GIFT-Eval rolling test windows (`exact_gift_eval=True`) reproduced the published GIFT-Eval leaderboard MASE to 6 decimal places (`abs_diff = 0.000000`):
     - `us_births/M/short`: local `0.760531` vs leaderboard `0.760531` (`diff = 0.000000`)
     - `saugeen/W/short`: local `1.990658` vs leaderboard `1.990658` (`diff = 0.000000`)
     - `solar/W/short`: local `1.470351` vs leaderboard `1.470351` (`diff = 0.000000`)
     - `hospital/M/short`: local `0.920528` vs leaderboard `0.920528` (`diff = 0.000000`)
   - *Key Factor:* Matching GluonTS `get_seasonality(freq)` (`m = 1` for `D` and `W`, `m = 12` for `M`, `m = 4` for `Q`, `m = 24` for `H`) and `LastValueImputation()` on training series was essential to achieve zero discrepancy.
9. **Batched `StatsForecast` Execution & `season_length` Cap for `AutoETS`/`AutoARIMA`:**
   - *Context:* `StatsForecast`'s `AutoETS` supports `season_length <= 24`, and state-space `AutoARIMA` with `m = 52` creates large companion matrices.
   - *Decision:* In `run_classical_on_dataset()`, all windows of a configuration are batched into a single vectorized `StatsForecast.forecast()` call, and `season_length` is passed as `m` when `m <= 24` (and `1` when `m > 24`) for `AutoETS` and `AutoARIMA`, while `SeasonalNaive` and `AutoTheta` (which uses classical seasonal decomposition) retain the full `m` (e.g. `52` on `unseen_mandi/W/short`).

## 2026-10-16 — Phase 3: Meta-Features and Pretraining Domain Similarity

10. **Meta-Feature Extraction Across All 97 GIFT-Eval Configs + 2 Unseen Configs (`src/tsdomain/features/run_features.py`):**
    - *Context:* To maximize statistical power in Phase 6 mixed-effects models (`M0 → M1 → M2 → M3`), we need configuration-level meta-features across all 97 GIFT-Eval configurations as well as both unseen datasets.
    - *Decision:* Download and cache all 45 GIFT-Eval Arrow subdirectories $\le 16.5\text{ MB}$ (covering 20 of the 23 base datasets directly from raw series). For high-frequency configurations whose raw Arrow shard exceeds $16.5\text{ MB}$ (e.g., `electricity/15T` at $594\text{ MB}$, `bitbrains_fast_storage/5T` at $288\text{ MB}$), compute features from the same base dataset's cached hourly/daily shard (`base_dataset_shard`) or domain proxy (`domain_proxy_shard`) and record `feature_source` explicitly in `dataset_features.parquet`.
11. **Leave-One-Base-Dataset-Out Reference Pool in `sim_distance` (`src/tsdomain/features/similarity.py`):**
    - *Context:* If an in-domain test dataset $d$ is included in its own reference pool $\mathcal{R}_m$, its distance to the nearest reference dataset is trivially zero, artificially inflating the separation between in-domain and out-of-domain datasets.
    - *Decision:* When computing continuous `sim_distance` (regularized Mahalanobis distance with ridge $\lambda = 0.25I$) and `sim_distance_knn` ($k = 3$ nearest reference base datasets), we exclude all configurations belonging to the test dataset's own `base_dataset` from $\mathcal{R}_m$.

## 2026-10-16 / 2026-10-19 — Phase 4: Zero-Shot TSFM Inference & Compute Profiling

12. **Unified PyTorch Environment & `jaxtyping.PyTree` Shim for `uni2ts` (`src/tsdomain/models/moirai_cli.py`):**
    - *Context:* `chronos-forecasting==2.3.2`, `uni2ts==2.0.0`, `timesfm==3.0.2`, and `granite-tsfm==0.3.10` all run cleanly in the same Python 3.11 + `torch==2.14.1+cpu` environment when installed with `--no-deps`, except that `uni2ts` imports `from jaxtyping import Float, PyTree`, which in `jaxtyping==0.2.34` raises `ImportError` (or in `0.3.11` imports `jax.tree_util`).
    - *Decision:* Provide a 5-line subscriptable `_PyTreeShim` on `jaxtyping.PyTree` before importing `uni2ts`, avoiding a redundant `jax` installation while preserving full `MoiraiModule` / `MoiraiForecast` execution.
13. **`TimesFM` PyTorch Safetensors Checkpoint & In-Place Input List Mutation Fix (`src/tsdomain/models/timesfm_cli.py`):**
    - *Verified Fact:* `google/timesfm-1.0-200m-pytorch` distributes legacy `.ckpt` weights, whereas `timesfm==3.0.2` (`TimesFM_2p5_200M_torch`) natively loads `google/timesfm-2.5-200m-pytorch` (`model.safetensors`, $231.3\text{M}$ parameters) with continuous quantile heads (`q0.1..q0.9`). Furthermore, `timesfm_2p5_base.py` lines 167–168 mutate the caller's `inputs` list in-place (`inputs += ...`) when `len(inputs) % global_batch_size != 0`.
    - *Decision:* Update `configs/models.yaml` to `google/timesfm-2.5-200m-pytorch` (`torch_compile=False` on Windows CPU) and pass a shallow copy `inputs=list(clean_inputs)` in `TimesFMAdapter.predict_batch()`.

## 2026-10-16 / 2026-10-19 — Phase 5–9: Conformal Calibration, Mixed Models, Finance Breaks & Ablations

14. **Crossed Random Effects via Variance Components in `statsmodels.MixedLM` (`src/tsdomain/analysis/mixed_models.py`):**
    - *Context:* Phase 6 requires crossed random intercepts $(1 \mid \text{base\_dataset}) + (1 \mid \text{model\_family})$ across $N = 1,463$ `(model, config)` pairs.
    - *Decision:* Use a single top-level group with explicit variance components `vc_formula={"base_dataset": "0 + C(base_dataset)", "model_family": "0 + C(model_family)"}`, fit with ML (`reml=False`) for nested likelihood-ratio tests (`M0 -> M1 -> M2a/M2b/M2 -> M3`), and refit with REML (`reml=True`) plus 1,000-cluster bootstrap on `base_dataset` for final parameter CIs.
15. **Time-Ordered Split Conformal Calibration (`src/tsdomain/metrics/conformal.py`):**
    - *Decision:* Strictly partition rolling evaluation windows chronologically (`earliest 20% -> calibration`, `latest 80% -> evaluation`) and apply Conformalized Quantile Regression (CQR; Romano et al., 2019) to quantile-emitting models and absolute-residual split conformal prediction to point-only models (`TTM-R2-Pretrained`, `Naive`, `Drift`, `LightGBM`).
16. **S&P 500 Volatility PELT & BOCPD Changepoint Alignment (`src/tsdomain/analysis/finance_case.py`, `src/tsdomain/analysis/ablations.py`):**
    - *Decision:* Apply `ruptures.Pelt(model="l2", min_size=15, jump=1)` with penalty calibrated to major regime shifts (`pen=12.0`) on 5-day smoothed S&P 500 Garman–Klass log-volatility (recovering 5 of 6 labelled crisis dates within $\pm 30$ trading days), and cross-check against Bayesian Online Changepoint Detection (BOCPD; Adams & MacKay, 2007) in Phase 8b.4.

