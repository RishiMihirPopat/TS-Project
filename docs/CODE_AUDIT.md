# Comprehensive Code Audit: `tsdomain` Benchmark Repository

**Audit Date:** 2026-10-19 (Updated & Resolved: 2026-11-04)  
**Scope:** All 22 Python modules in `src/tsdomain/`, configuration files in `configs/`, test suite in `tests/`, and generated artifacts in `results/` and `report/`.

---

## Executive Summary

The codebase is modular, deterministic, and passes all unit/acceptance tests in `tests/`. A line-by-line audit of the implementation against the statistical and methodological specifications in the Build Plan and Ponytail standards identified and **resolved all 10 concrete issues** plus dead-code/duplication cleanups:
- **4 High-Severity (Methodological / Statistical Bugs — RESOLVED)** affecting split-conformal calibration across multi-scale series (#1.1), cluster bootstrap confidence intervals (#1.2), meta-feature extraction for large Arrow shards (#1.3), and LightGBM validation splitting (#1.4).
- **4 Medium-Severity (Registry, Profiling & Case-Study Bugs — RESOLVED)** involving model-name lookup mismatches in the mixed-effects regression (#2.1), hardcoded leakage flags (#2.2), synthetic compute profiling repeats/TDP constants (#2.3), and post-break origin indexing in the finance case study (#2.4).
- **2 Low-Severity (Documentation & Narrative Drift — RESOLVED)** where generated report prose and handoff docs diverged slightly from code constants (#3.1, #3.2).
- **Ponytail Codebase Hygiene (RESOLVED):** Removed 350+ lines of duplicate window-scoring loops (`score_forecast_window` in `src/tsdomain/utils.py`), unused profiler types (`ComputeProfileResult`, `profile_callable_repeats`), duplicate loss/test functions (`qlike_loss`, `kupiec_pof_test`), duplicate CLI dispatch branches, and hardcoded config/registry dictionaries; added `# shortcut: <limit>, <when to upgrade>` annotations on all intentional runtime/offline shortcuts; added strict out-of-fold standardization in `grouped_cv.py`.

---

## 1. High-Severity Findings (Methodological & Statistical)

### 1.1 Unnormalized Multi-Series Residual Pooling & Single-Window Splitting in Split Conformal Calibration
- **Location:** [`src/tsdomain/metrics/conformal.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/conformal.py#L37-L137) and [`run_conformal_protocol`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/conformal.py#L200-L252)
- **What the code does:**
  1. [`apply_conformal_to_forecast_df`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/conformal.py#L74-L137) pools raw, unscaled nonconformity scores $s_i = \max(\hat{q}_{0.1,i} - y_i,\; y_i - \hat{q}_{0.9,i})$ (CQR, lines 44–53) or $|y_i - \hat{\mu}_i|$ (residual conformal, lines 63–69) across **all `item_id`s** in `cal_df` into a single scalar `offset` in raw units (lines 106, 127) and adds `offset` to every series in `eval_df` (lines 119, 134).
  2. When `len(unique_win_ids) == 1` (e.g., `m4_monthly`, `m4_quarterly`), lines 91–95 sort by `["timestamp", "item_id", "horizon_step"]` and use the first 20% of rows as calibration and the remaining 80% as evaluation—splitting across series and horizon steps within the same forecast window rather than calibrating on historical windows.
  3. Lines 211–216 call [`coverage_and_width`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/probabilistic.py#L81-L115) on the pooled `pre_df` / `post_df` without `y_train`, which normalizes interval width by the pooled `mean(abs(y_true))` across all series.
- **Impact:** On heterogeneous multi-series datasets such as `unseen_recency/D/short` (which mixes temperature in `°C` [~20] with solar radiation `shortwave_radiation_sum` in `Wh/m²` [~20,000]), the raw `offset` is dominated by the solar radiation series (~thousands of units) and is added directly to temperature series, inflating normalized interval width after calibration.
- **Recommended Fix:** Scale nonconformity scores per series by each series' in-sample scale (e.g., seasonal naive error or $\text{mean}(|y_{\text{train}}|)$) before pooling, or calibrate per `item_id`, and compute [`coverage_and_width`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/probabilistic.py#L81-L115) per series before averaging.

---

### 1.2 Pooled OLS Cluster Bootstrap vs. REML MixedLM Point Estimates
- **Location:** [`cluster_bootstrap_fixed_effects`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py#L315-L345) in [`src/tsdomain/analysis/mixed_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py)
- **What the code does:**
  - Point estimates (`estimate_reml`) and analytic standard errors (`se_reml`) are estimated via [`_fit_crossed_mixedlm`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py#L266-L312), which fits a linear mixed-effects model with crossed random intercepts $(1 \mid \text{base\_dataset}) + (1 \mid \text{model\_family})$.
  - However, inside [`cluster_bootstrap_fixed_effects`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py#L333), each bootstrap replicate fits plain pooled **OLS** (`smf.ols(formula, data=boot_df).fit()`) without random intercepts.
- **Impact:** Because `base_dataset` clusters have unbalanced sizes (from 1 config in `hospital`/`covid_deaths` to 9 configs in `electricity`/`solar`), pooled OLS weights large clusters 9× more heavily than single-config clusters, whereas REML partially pools cluster means according to the estimated variance components. Consequently, in [`results/analysis/mixed_model_coefficients.csv`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/results/analysis/mixed_model_coefficients.csv), several `estimate_reml` values fall outside `[boot_ci95_low, boot_ci95_high]` (for example, in `M2`, `Intercept` has `estimate_reml = 0.2358` vs. `boot_ci95 = [-0.4916, 0.1949]`, and `z_log_params` has `estimate_reml = -0.0005` vs. `boot_ci95 = [0.0281, 0.0725]`).
- **Recommended Fix:** Fit `_fit_crossed_mixedlm(boot_df, formula, reml=True)` inside each bootstrap replicate (using a new cluster ID per resampled `base_dataset` draw so duplicated clusters remain distinct random-effect groups), or weight OLS by inverse cluster size when computing fallback CIs.

---

### 1.3 Uncalled HuggingFace `datasets-server` URL & Silent `domain_proxy_shard` Duplication
- **Location:** [`_load_series_for_gift_config`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/features/run_features.py#L108-L130) in [`src/tsdomain/features/run_features.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/features/run_features.py)
- **What the code does:**
  - When a GIFT-Eval dataset's Arrow shard exceeds `max_shard_mb = 16.5` and no smaller shard exists for the same `base_dataset`, lines 108–115 import `requests` and construct `url = "https://datasets-server.huggingface.co/rows?..."`, **never execute `requests.get(url)`**, and immediately fall through to lines 117–126 (`feature_source = "domain_proxy_shard"`), which copies `series_list` from the smallest shard in the same `domain`.
- **Impact:** Any large-shard dataset without a small sibling shard gets exact duplicate meta-features (`spectral_entropy`, `hurst`, `trend_strength`, `seasonal_strength`, `lumpiness`, `stability`) copied from another dataset in its domain, artificially reducing within-domain meta-feature variance in [`results/features/dataset_features.parquet`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/results/features/dataset_features.parquet).
- **Recommended Fix:** Stream only the first $N=50$ series from the Arrow shard via HTTP Range request or `pyarrow` / HuggingFace streaming (or actually query the `datasets-server` / parquet endpoint), and remove the silent `domain_proxy_shard` copy.

---

### 1.4 Shuffled Train/Validation Split & Full-Series Z-Scoring in LightGBM Baseline
- **Location:** [`_build_lag_features`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/lgbm.py#L31-L68) and [`fit_predict_lgbm_dataset`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/lgbm.py#L71-L162) in [`src/tsdomain/baselines/lgbm.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/lgbm.py)
- **What the code does:**
  1. In [`_build_lag_features`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/lgbm.py#L54-L56), each window's `y_train` is z-scored over its entire length (`mu = np.mean(y)`, `sd = np.std(y)`), and when a dataset has 2 rolling windows (`w=0` and `w=1`), overlapping lag windows from the same series are pooled into `X_all`.
  2. When `len(X_all) > max_train_samples` (lines 90–93), `rng.choice(len(X_all), size=max_train_samples, replace=False)` randomly shuffles rows across series and timestamps before splitting `X_tr = X_all[:-n_val]`, `X_va = X_all[-n_val:]` (lines 96–97).
- **Impact:** The validation set used to select `num_leaves` contains rows from the exact same series and overlapping rolling windows as `X_tr`. While `y_test` itself is never leaked into `y_train`, temporal/window overlap between `X_tr` and `X_va` favors higher-capacity trees (`num_leaves = 63`).
- **Recommended Fix:** Reserve only the last temporal block of each series' earliest training window for validation (prior to subsampling), and avoid pooling overlapping `w=0` and `w=1` histories into the training matrix.

---

## 2. Medium-Severity Findings (Registry, Profiling & Case Study)

### 2.1 Model ID Lookup Mismatches in `mixed_models.py` (`Moirai2`, `timesfm_2_0_500m`, `TTM-R1-Pretrained`)
- **Location:** [`build_analysis_panel`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py#L110-L161) in [`src/tsdomain/analysis/mixed_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py)
- **What the code does:**
  - [`results/leaderboard/leaderboard.parquet`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/results/leaderboard/leaderboard.parquet) and [`configs/pretraining_registry.yaml`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/configs/pretraining_registry.yaml) use the canonical model identifiers `"Moirai2"`, `"timesfm_2_0_500m"`, and `"TTM-R1-Pretrained"`.
  - However, `param_map` (lines 110–128) and `family_map` (lines 129–145) in [`src/tsdomain/analysis/mixed_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py) key on `"Moirai-2.0-R-small"` and `"TimesFM-2.0"`, and omit `"TTM-R1-Pretrained"`.
  - Furthermore, the fallback on line 156 maps `"Moirai2"` to family `"moirai2"`, whereas `domain_share_by_family` on line 151 keys on `"moirai_2"`.
- **Impact:**
  1. `"Moirai2"`, `"timesfm_2_0_500m"` (actually 500M parameters), and `"TTM-R1-Pretrained"` (actually ~1M parameters) all fall back to the default `param_count = 50,000,000` (`50.00M`), distorting `z_log_params` in the mixed-effects models and displaying `50.00M` in [`report/tables/table2_pretraining_registry.md`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/report/tables/table2_pretraining_registry.md).
  2. `"Moirai2"` fails the `"moirai_2"` lookup in `domain_share_by_family` and receives a flat fallback `sim_share = 0.05` across all domains.
- **Recommended Fix:** Read `param_count`, `model_family`, and `domain_share` directly from [`configs/pretraining_registry.yaml`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/configs/pretraining_registry.yaml) and [`results/features/similarity_matrix.parquet`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/results/features/similarity_matrix.parquet) instead of maintaining duplicate hardcoded dictionaries in `mixed_models.py`.

---

### 2.2 Hardcoded `leak_bases` Set & Hardcoded Ablation Conclusion String
- **Location:** [`build_analysis_panel`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py#L187-L190) in [`src/tsdomain/analysis/mixed_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py) and [`run_leakage_exclusion_ablation`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/ablations.py#L254-L261) in [`src/tsdomain/analysis/ablations.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/ablations.py)
- **What the code does:**
  - Even though [`src/tsdomain/data/leaderboard.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/leaderboard.py#L118) extracts the per-model `test_data_leakage_flag` from GIFT-Eval's official metadata, `mixed_models.py` line 187 overwrites `in_pretraining_pool` with a static 6-dataset set `{"ett1", "ett2", "solar", "saugeen", "kdd_cup_2018", "electricity"}` for all models.
  - In [`ablations.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/ablations.py#L258), the `conclusion` column is hardcoded to `"Domain-similarity block remains statistically significant after dropping overlap datasets"` regardless of the computed `p_m2_vs_m1` value (which is `0.0906` in [`results/analysis/ablation_leakage_exclusion.csv`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/results/analysis/ablation_leakage_exclusion.csv)).
- **Recommended Fix:** Combine the model-specific `test_data_leakage_flag` and registry `in_pretraining_pool` from `similarity_matrix.parquet`, and generate the ablation conclusion string dynamically from `p_m2_vs_m1 < 0.05`.

---

### 2.3 Compute Profiler CodeCarbon Bypass, TDP Inconsistency & Synthetic Repeats
- **Location:**
  - [`profile_callable`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/compute/profiler.py#L214-L263) in [`src/tsdomain/compute/profiler.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/compute/profiler.py)
  - [`_ingest_classical_compute_logs`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/run_models.py#L46-L90) and [`run_phase6_and_7`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/run_models.py#L145-L176) in [`src/tsdomain/models/run_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/run_models.py)
  - [`run_classical_baselines_on_dataset`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/classical.py#L401) in [`src/tsdomain/baselines/classical.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/classical.py)
- **What the code does:**
  1. `profile_callable()` defaults to `use_codecarbon: bool = False` (line 218), and no caller passes `use_codecarbon=True`. Consequently, every energy number in [`results/compute/compute_profile.parquet`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/results/compute/compute_profile.parquet) is computed via the TDP fallback `energy_kwh = wall_clock_s * 45.0 / 3,600,000`.
  2. Because energy is a fixed scalar multiple of runtime (`12.5 mWh/s`) for every model, the latency Pareto frontier and energy Pareto frontier in [`src/tsdomain/analysis/pareto.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/pareto.py) are identical up to a constant factor.
  3. [`classical.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/classical.py#L401) logs `energy_kwh` using `15.0 W`, while [`run_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/run_models.py#L71) overwrites it using `45.0 W` and synthesizes 3 repeats for classical baselines using deterministic `jitter = 1.0 + 0.015 * (j - 1)` (line 69) instead of timing 3 actual runs.
  4. For `Chronos_small`, repeats 2 and 3 in [`run_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/run_models.py#L151-L155) run on `windows[:20]` and linearly extrapolate runtime/energy by `n_pts / n_pts_sub`.
- **Recommended Fix:** Harmonize the TDP constant across `classical.py` and `profiler.py`, document explicitly in `energy_method` when sub-window scaling or single-run logging is used, and avoid deterministic synthetic jitter on repeat logs.

---

### 2.4 Finance Case Study Post-Break Origin Selection
- **Location:** [`run_finance_case_study`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/finance_case.py#L228-L260) in [`src/tsdomain/analysis/finance_case.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/finance_case.py)
- **What the code does:**
  - [`detect_spx_volatility_breaks`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/finance_case.py#L51-L130) runs PELT (`ruptures.Pelt(model="rbf")`) on the full 1990–2026 log-volatility series and matches each macro crisis to its nearest PELT change-point (`detected_break_idx`).
  - While `is_post_break` (lines 236–238) marks `[b_idx, b_idx + 60]` around **both** `labelled_trading_idx` and `detected_break_idx`, `post_break_origins` (lines 246–250) generates evaluation origins only at offsets `[2, 10, 22, 38]` after `labelled_trading_idx` (ignoring `detected_break_idx`).
- **Recommended Fix:** Anchor `post_break_origins` to the PELT-detected breakpoints (`detected_break_idx`) or evaluate both sets explicitly so the PELT change-point detection directly drives the post-break evaluation windows.

---

## 3. Low-Severity Findings (Documentation & Narrative Drift)

### 3.1 Hardcoded Station & Variable Counts in `report_assets.py` vs. `unseen_recency.py`
- **Location:** [`generate_report_markdown`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/report_assets.py#L646-L647) in [`src/tsdomain/analysis/report_assets.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/report_assets.py) vs. [`src/tsdomain/data/unseen_recency.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/unseen_recency.py#L22-L49) and [`src/tsdomain/data/unseen_mandi.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/unseen_mandi.py#L40-L43)
- **Details:**
  - `report_assets.py` line 647 states that `unseen_recency/D/short` contains *"5 meteorological variables ... across 12 global stations"* (60 series), whereas [`unseen_recency.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/unseen_recency.py#L22-L49) actually fetches **3 variables** (`temperature_2m_mean`, `shortwave_radiation_sum`, `wind_speed_10m_max`) across **20 global stations** ($20 \times 3 = 60$ series).
  - [`unseen_mandi.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/unseen_mandi.py#L43) comments state `260 weeks` (`2012-01-02` to `2016-12-26`), whereas `pd.date_range("2012-01-02", "2016-12-26", freq="W-MON")` produces **261 weekly Mondays**.

### 3.2 Forecast Parquet Column Naming in `docs/AGENT_HANDOFF_REFERENCE.md`
- **Location:** [`docs/AGENT_HANDOFF_REFERENCE.md`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/docs/AGENT_HANDOFF_REFERENCE.md#L168-L180) vs. [`FORECAST_COLUMNS`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/utils.py#L17-L25) in [`src/tsdomain/utils.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/utils.py)
- **Details:**
  - `FORECAST_COLUMNS` in [`src/tsdomain/utils.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/utils.py#L17-L25) defines the canonical columns as `["item_id", "window_id", "horizon_step", "timestamp", "y_true", "mean", "q0.1", ..., "q0.9"]`, whereas Section 4.2 of `docs/AGENT_HANDOFF_REFERENCE.md` listed `series_id`, `step`, and `y_hat`.

---

## Summary Table of Audited Modules

| Module | Status | Findings & Resolution |
|---|---|---|
| [`src/tsdomain/config.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/config.py) | Clean | Added `param_count` to `PretrainingModelEntry` |
| [`src/tsdomain/utils.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/utils.py) | Clean | Added shared `score_forecast_window()`; aliased `detect_device = resolve_device` |
| [`src/tsdomain/metrics/point.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/point.py) | Clean | Exact MASE, sMAPE, RMSSE, Geometric Mean Relative MASE |
| [`src/tsdomain/metrics/probabilistic.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/probabilistic.py) | Clean | Exact 9-quantile CRPS & Winkler score |
| [`src/tsdomain/metrics/conformal.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/metrics/conformal.py) | **Resolved (#1.1)** | Per-series scale-normalized CQR/residual scores & within-series chronological `w=1` split |
| [`src/tsdomain/baselines/classical.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/classical.py) | **Resolved (#2.3)** | Uses `DEFAULT_CPU_TDP_WATTS` (`45.0 W`) and shared `score_forecast_window()` |
| [`src/tsdomain/baselines/lgbm.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/baselines/lgbm.py) | **Resolved (#1.4)** | Earliest-window-only lag features, pre-validation z-scoring, temporal train/val split |
| [`src/tsdomain/data/gift_eval.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/gift_eval.py) | Clean | — |
| [`src/tsdomain/data/leaderboard.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/leaderboard.py) | Clean | Properly ingests official GIFT-Eval CSVs + `config.json` |
| [`src/tsdomain/data/unseen_mandi.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/unseen_mandi.py) | **Resolved (#3.1)** | Fixed comment to 261 weeks |
| [`src/tsdomain/data/unseen_recency.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/unseen_recency.py) | Clean | Fetches 20 stations × 3 vars = 60 series from Open-Meteo |
| [`src/tsdomain/data/finance.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/data/finance.py) | Clean | — |
| [`src/tsdomain/features/meta_features.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/features/meta_features.py) | Clean | STL, Hurst R/S, Welch spectral entropy, lumpiness, stability |
| [`src/tsdomain/features/similarity.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/features/similarity.py) | **Resolved (#2.1)** | Propagates canonical `model_family` and `param_count` from `pretraining_registry.yaml` |
| [`src/tsdomain/features/run_features.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/features/run_features.py) | **Resolved (#1.3)** | Removed uncalled URL & silent copy; added deterministic `domain_pooled_sample` fallback |
| [`src/tsdomain/models/chronos_cli.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/chronos_cli.py) | Clean | Real `amazon/chronos-t5-small` & `chronos-bolt-small` inference |
| [`src/tsdomain/models/moirai_cli.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/moirai_cli.py) | Clean | Real `Salesforce/moirai-1.1-R-small` via `uni2ts` |
| [`src/tsdomain/models/timesfm_cli.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/timesfm_cli.py) | Clean | Real `google/timesfm-2.5-200m-pytorch` inference |
| [`src/tsdomain/models/ttm_cli.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/ttm_cli.py) | Clean | Real `ibm-granite/granite-timeseries-ttm-r2` inference |
| [`src/tsdomain/models/run_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/models/run_models.py) | **Resolved (#2.3)** | Removed synthetic jitter on cached classical repeats; tagged `energy_source` on sub-window scaling |
| [`src/tsdomain/compute/profiler.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/compute/profiler.py) | **Resolved (#2.3)** | Removed dead `ComputeProfileResult`/`profile_callable_repeats`; annotated TDP shortcut |
| [`src/tsdomain/analysis/mixed_models.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/mixed_models.py) | **Resolved (#1.2, #2.1, #2.2)** | Henderson MixedLM GLS cluster bootstrap; registry-driven family/params; combined leakage flag |
| [`src/tsdomain/analysis/grouped_cv.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/grouped_cv.py) | **Resolved** | Added strict out-of-fold standardization of `z_*` predictors inside each LODO fold |
| [`src/tsdomain/analysis/finance_case.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/finance_case.py) | **Resolved (#2.4)** | Post-break origins anchored to PELT `detected_break_idx`; removed duplicate `qlike_loss`/`kupiec_pof_test` |
| [`src/tsdomain/analysis/ablations.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/ablations.py) | **Resolved (#2.2)** | Dynamic leakage-exclusion conclusion string from computed $p$-values |
| [`src/tsdomain/analysis/pareto.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/pareto.py) | Clean | — |
| [`src/tsdomain/analysis/report_assets.py`](file:///c:/Users/rishi/Documents/antigravity/quick-raman/src/tsdomain/analysis/report_assets.py) | **Resolved (#3.1)** | Fixed narrative text to 3 variables across 20 global stations |
