# Codebase Walkthrough (`docs/walkthrough.md`)

This document explains every module in `src/tsdomain/` and `seed/` in plain language, including its purpose, design rationale, and the exact mathematical formulas or literature references it implements (Working Rule 5).

---

## Phase 0 — Bootstrap, Baseline Harness, and Metrics

### 1. `seed/baseline_model.py`
- **What it does:** Evaluates the five mandatory classical baselines (`Seasonal Naive`, `Drift`, `ETS(A_d, M)`, `Theta`, and `SARIMA(1,1,1)(1,1,1)_12`) on the classic Box & Jenkins `AirPassengers` monthly series (`n_train = 120`, `n_test = 24`, season length $m = 12$).
- **Why it exists:** Serves as the immutable regression anchor from Section 3 of the Milestone 1 report, ensuring our metric computations and classical model wrappers match the baseline report before scaling to GIFT-Eval and unseen domains.
- **Formulas implemented:**
  - $\text{MASE} = \frac{\frac{1}{H}\sum_{t=1}^H |y_{T+t} - \hat{y}_{T+t}|}{\frac{1}{T-m}\sum_{s=m+1}^T |y_s - y_{s-m}|}$
  - $\text{RMSE} = \sqrt{\frac{1}{H}\sum_{t=1}^H (y_{T+t} - \hat{y}_{T+t})^2}$
  - $\text{MAPE} = \frac{100}{H}\sum_{t=1}^H \left|\frac{y_{T+t} - \hat{y}_{T+t}}{y_{T+t}}\right|$

### 2. `src/tsdomain/metrics/point.py` & `src/tsdomain/metrics/probabilistic.py`
- **What they do:** Implement all point and probabilistic evaluation metrics specified in Section 5 of the Build Plan.
- **Why they exist:** Centralizing metric definitions guarantees that classical baselines, local TSFM runs, and GIFT-Eval cross-checks use identical scaling, quantile weighting, and aggregation rules.
- **Formulas implemented:**
  - **MASE & RMSSE:** Hyndman & Koehler (2006) seasonal naive in-sample scaling, falling back to $m=1$ when training length $T < 2m$.
  - **Pinball Loss:** $\rho_\tau(y, \hat{q}_\tau) = \max(\tau(y - \hat{q}_\tau), (\tau - 1)(y - \hat{q}_\tau))$ for $\tau \in \{0.1, \dots, 0.9\}$.
  - **CRPS (Weighted Quantile Loss approximation):** Matches GIFT-Eval (Aksu et al., 2024) `mean_weighted_sum_quantile_loss`:
    $$\text{CRPS} \approx \frac{2}{|Q|} \sum_{\tau \in Q} \frac{\sum_{t=1}^H \rho_\tau(y_t, \hat{q}_{t,\tau})}{\sum_{t=1}^H |y_t|}$$
  - **Empirical Coverage & Scaled Interval Width:** Fraction of $y_t \in [\hat{q}_{0.1}, \hat{q}_{0.9}]$ and mean width divided by mean absolute training value.
  - **Winkler Interval Score ($\alpha = 0.2$):** $(u - l) + \frac{2}{\alpha}(l - y)\mathbb{I}[y < l] + \frac{2}{\alpha}(y - u)\mathbb{I}[y > u]$.
  - **QLIKE (Patton, 2011):** $\frac{\sigma^2_{\text{true}}}{\hat{\sigma}^2} - \ln\left(\frac{\sigma^2_{\text{true}}}{\hat{\sigma}^2}\right) - 1$.
  - **Response Variable (`log_ratio`):** $\ln(\text{MASE}_{\text{model}} / \text{MASE}_{\text{best\_classical}})$, where negative values indicate TSFM outperformance.

### 3. `src/tsdomain/baselines/classical.py`
- **What it does:** Wraps both `statsforecast` (`SeasonalNaive`, `Naive`, `RandomWalkWithDrift`, `AutoETS`, `AutoTheta`, `AutoARIMA`) and the exact `statsmodels` reference specifications (`ETS(A_d,M)`, `Theta(12)`, `SARIMA(1,1,1)(1,1,1)_12`) behind a unified rolling-window interface that outputs canonical forecast `.parquet` files and `.meta.json` sidecars.
- **Why it exists:** Provides the mandatory classical denominator ($\text{MASE}_{\text{best\_classical}}$) across all benchmark datasets while maintaining exact regression compatibility with Phase 0.

---

## Phase 1 — Data Acquisition, Leaderboard Ingestion, and Registry

### 4. `src/tsdomain/config.py`
- **What it does:** Validates `configs/datasets.yaml`, `configs/models.yaml`, `configs/pretraining_registry.yaml`, and `configs/experiment.yaml` using Pydantic v2 models and resolves `inherits_from` links across model family checkpoints.
- **Why it exists:** Prevents silent typos in dataset identifiers, horizons, or domain coverage claims.

### 5. `src/tsdomain/data/leaderboard.py`
- **What it does:** Downloads `all_results.csv` and `config.json` for 24 statistical, deep-learning (`PatchTST`, `N-BEATS`, `DLinear`, `TFT`), and foundation models (`Chronos`, `Chronos-Bolt`, `Chronos-2`, `Moirai`, `Moirai2`, `TimesFM`, `TTM`) from `SalesforceAIResearch/gift-eval` and consolidates them into `results/metrics/leaderboard.parquet`.
- **Why it exists:** Supplies the 97-configuration benchmark results and `test_data_leakage_flag` indicators required for Phase 2 cross-checks, Phase 6 hierarchical modeling, and Phase 8 leakage ablations.

### 6. `src/tsdomain/data/gift_eval.py`
- **What it does:** Downloads Hugging Face Arrow shards from `Salesforce/GiftEval`, flattens multivariate series into univariate `item_id_dim{k}` series (`MultivariateToUnivariate`), and constructs rolling evaluation windows matching `src/gift_eval/data.py` (`TEST_SPLIT = 0.1`, `MAX_WINDOW = 20`).
- **Why it exists:** Ensures local classical and TSFM evaluations use the exact GIFT-Eval horizon and windowing rules.

### 7. `src/tsdomain/data/unseen_mandi.py`, `unseen_recency.py`, & `finance.py`
- **What they do:** Build the two unseen benchmark datasets (`unseen_mandi` with 60 weekly Indian agricultural mandi price series over 2012–2016; `unseen_recency` with 60 daily post-cutoff 2025–2026 meteorological series) and the S&P 500 daily OHLC + Garman–Klass volatility dataset (`spx_daily.parquet`, 1990–2026).
- **Formulas implemented:**
  - **Garman–Klass (1980) daily variance:**
    $$\sigma^2_{\text{GK}, t} = \frac{1}{2}\left(\ln\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_t}{O_t}\right)^2$$

---

## Phase 2 — Classical Baselines & Leaderboard Cross-Check

### 8. `src/tsdomain/baselines/run_baselines.py`
- **What it does:**
  1. Executes `run_leaderboard_crosscheck()` on full GIFT-Eval configurations to verify that our local windowing and MASE computation match the published GIFT-Eval `Seasonal_Naive` leaderboard exactly (`abs_diff < 1e-6`).
  2. Computes `best_classical` across all 97 GIFT-Eval leaderboard configurations (`results/metrics/best_classical_leaderboard.parquet`) as $\min(\text{MASE}_{\text{Seasonal\_Naive}}, \text{MASE}_{\text{Auto\_ETS}}, \text{MASE}_{\text{Auto\_Theta}}, \text{MASE}_{\text{Auto\_Arima}})$.
  3. Runs `Naive`, `SeasonalNaive`, `Drift`, `AutoETS`, `AutoTheta`, and `AutoARIMA` across the 13 local benchmark configurations (11 stratified GIFT-Eval configs + `unseen_mandi/W/short` + `unseen_recency/D/short`), writing canonical forecast parquets (`results/forecasts/{model}/{config}.parquet`), `.meta.json` sidecars, and `results/metrics/best_classical_local.parquet`.
- **Why it exists:** Establishes the verified classical baseline denominator ($\text{MASE}_{\text{best\_classical}}$) required for `log_ratio` in Phase 6 and Pareto efficiency in Phase 9.

---

## Phase 3 — Meta-Features and Domain Similarity

### 9. `src/tsdomain/features/meta_features.py`
- **What it does:** Computes per-series structural meta-features and aggregates them via median to configuration level (`results/features/dataset_features.parquet`).
- **Formulas implemented:**
  - **STL Trend & Seasonal Strength (Hyndman & Athanasopoulos, FPP3 §4.3):**
    $$F_T = \max\left(0, 1 - \frac{\text{Var}(R_t)}{\text{Var}(T_t + R_t)}\right), \quad F_S = \max\left(0, 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_t + R_t)}\right)$$
  - **Normalized Spectral Entropy & Forecastability (Goerg, 2013):**
    $$H_s(x) = -\frac{1}{\log_2 K}\sum_{k=1}^K P(\omega_k)\log_2 P(\omega_k) \in [0, 1], \quad \text{forecastability} = 1 - H_s(x)$$
  - **Hurst Exponent (`hurst`):** Rescaled range ($R/S$) regression slope $\log(R/S)_k \sim H \log(k)$ across log-spaced window sizes $k$.
  - **Stability & Lumpiness (`tsfeatures`):** Variance of non-overlapping window means ($\text{stability}$) and window variances ($\text{lumpiness}$) on z-scored series.

### 10. `src/tsdomain/features/similarity.py` & `run_features.py`
- **What they do:** Map each `(model, config)` pair to categorical pretraining exposure (`domain_in_pretraining` $\in \{\text{true}, \text{false}, \text{unknown}\}$, plus optimistic/conservative codings) and continuous standardized feature-space distance (`sim_distance` via regularized Mahalanobis distance and `sim_distance_knn` via 3-NN Euclidean distance, leaving out the target `base_dataset`), alongside correlation and VIF diagnostics (`results/features/vif_report.csv`).

---

## Phase 4 — Zero-Shot TSFM Inference & Compute Profiling

### 11. `src/tsdomain/models/base.py`, `chronos_cli.py`, `moirai_cli.py`, `timesfm_cli.py`, `ttm_cli.py`, & `run_models.py`
- **What they do:** Implement the common preprocessing/z-score scaling protocol (`context_length = 512`, `apply_common_scaling`, `invert_common_scaling`) and zero-shot inference adapters for `Chronos_small` (`amazon/chronos-t5-small`), `Moirai_small` (`Salesforce/moirai-1.1-R-small`), `TimesFM` (`google/timesfm-2.5-200m-pytorch`), and `TTM-R2-Pretrained` (`ibm-granite/granite-timeseries-ttm-r2`).
- **Why they exist:** Ensure every TSFM sees identical context windows and scaling before measuring accuracy and compute cost across the 13 local configurations.

### 12. `src/tsdomain/compute/profiler.py`
- **What it does:** Measures model load time (`load_time_s`), total and per-point inference latency (`infer_time_ms_per_point`, `infer_time_iqr_ms`), peak host RSS (`peak_ram_mb`), peak GPU memory (`peak_vram_mb`), and energy (`energy_wh_total`, `energy_mwh_per_1k_points`, `energy_source`) across 3 profiled repeats after a 5-series warmup.

---

## Phase 5 — Uncertainty Quantification & Conformal Calibration (P1)

### 13. `src/tsdomain/metrics/conformal.py`
- **What it does:** Evaluates native prediction intervals (`q0.1..q0.9`) and applies time-ordered split conformal calibration (earliest 20% windows for calibration, remaining 80% for evaluation).
- **Formulas implemented:**
  - **Conformalized Quantile Regression (CQR; Romano et al., 2019):**
    $$s_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i,\; y_i - \hat{q}_{1-\alpha/2}(x_i)), \quad \hat{Q}_{1-\alpha} = \text{Quantile}_{(1-\alpha)(1 + 1/n_{\text{cal}})}(\{s_i\})$$
  - **Split Residual Conformal (for point-only models):** $s_i = |y_i - \hat{y}_i|$.

---

## Phase 6 — Hierarchical Mixed-Effects Models & LODO Cross-Validation

### 14. `src/tsdomain/analysis/mixed_models.py` & `grouped_cv.py`
- **What they do:** Fit the nested mixed-effects model ladder (`M0`, `M1`, `M2a`, `M2b`, `M2`, `M3`) with crossed random intercepts $(1 \mid \text{base\_dataset}) + (1 \mid \text{model\_family})$, compute Nakagawa & Schielzeth marginal/conditional $R^2$, 1,000-cluster bootstrap CIs, Benjamini–Hochberg FDR $q$-values, H2 strata contrasts, and Leave-One-Base-Dataset-Out (`LeaveOneGroupOut` on `base_dataset`) cross-validation against the Constant and Tan Jerome & Simon (2026) heuristics.

---

## Phase 7 — Finance / Volatility Structural-Break Case Study (P1)

### 15. `src/tsdomain/analysis/finance_case.py`
- **What it does:** Segments S&P 500 daily Garman–Klass log-volatility (1990–2026) via PELT (`ruptures`), verifies recovery of 6 historical crisis dates within $\pm 30$ trading days, and compares `Naive`, `EWMA(0.94)`, `GARCH(1,1)-t` (`arch`), `Chronos_small` (no finance in pretraining), and `Moirai_small` (finance in pretraining) across stable vs 63-day post-break windows using stationary block bootstrap CIs (Politis & Romano, 1994) and Kupiec (1995) POF interval tests.

---

## Phase 8 & Phase 9 — LightGBM, Robustness Ablations, Pareto Analysis & Report

### 16. `src/tsdomain/baselines/lgbm.py` & `src/tsdomain/analysis/ablations.py`
- **What they do:** Run a global autoregressive LightGBM baseline tuned with `optuna` under strict wall-clock budget parity with `Moirai_small`, and execute the four-part robustness ablation suite (common vs model-default scaling, pretraining leakage exclusion, `unknown` registry coding sensitivity, and BOCPD vs PELT break detection).

### 17. `src/tsdomain/analysis/pareto.py` & `report_assets.py`
- **What they do:** Identify the accuracy–energy and accuracy–latency Pareto frontiers, compute `energy_per_1pct_mase_gain_mwh`, map the H4 feature-space decision boundary, and programmatically render all 6 figures (`report/figures/`), 6 tables (`report/tables/`), and `report/report.md`.

