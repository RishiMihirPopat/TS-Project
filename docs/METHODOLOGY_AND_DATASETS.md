# Methodology and Dataset Description

**Project:** Domain-Shift and Computational-Efficiency Benchmarking of Time-Series Foundation Models (TSFMs)  
**Codebase:** `src/tsdomain/` (Python 3.11) | **Primary Report:** `report/report.md` | **Reproducibility Seed:** 42  

---

## 1. Research Question and Hypotheses

When a zero-shot Time-Series Foundation Model (TSFM) is evaluated on a domain differing from its documented pretraining corpus, how much of its claimed advantage over classical statistical baselines survives? Can transfer performance be predicted *ex ante* from measurable domain similarity and series characteristics? And when does the accuracy gain justify the 10–1,000$\times$ higher computational latency and energy cost?

| ID | Hypothesis Statement | Operational Statistical Contrast |
|---|---|---|
| **H1** | Pretraining domain similarity predicts zero-shot TSFM gain over the common classical baseline set beyond series difficulty characteristics (`forecastability`, `seasonal_strength`, `trend_strength`, `hurst`, `stability`, `lumpiness`, `log_length`) and test-data leakage. | Likelihood-Ratio Test ($\chi^2$) comparing nested mixed-effects models `M2` (with similarity block `sim_binary`, `sim_share`, `sim_distance`) vs. `M1` (difficulty + leakage controls) on the log-ratio response $y = \log(\mathrm{MASE}_{\text{model}} / \mathrm{MASE}_{\text{classical}})$. |
| **H2** | Zero-shot TSFM performance exhibits significant degradation in out-of-domain and post-cutoff temporal strata relative to in-domain benchmarks. | Planned cluster-robust contrasts across domain-coverage strata with $\ge 5$ base datasets (`in-domain`, `partial-overlap`, `out-of-domain`); strata with $< 5$ base datasets are evaluated descriptively only. |
| **H3** | Under structural volatility breaks on S&P 500 market data, a model with no daily market-price pretraining exposure (`Chronos_small`, finance via M4 macroeconomic series only) degrades more severely relative to stable regimes than a model pretrained on multi-domain web tables and market data (`Moirai_small`). | Difference-in-Differences (DiD) contrast: $(\bar{L}_{\text{Chronos, post-break}} - \bar{L}_{\text{Chronos, stable}}) - (\bar{L}_{\text{Moirai, post-break}} - \bar{L}_{\text{Moirai, stable}})$ evaluated on QLIKE, MSE, and MASE with Politis–Romano stationary block-bootstrap CIs and two-sided Diebold–Mariano tests. |
| **H4** | Across continuous feature space, classical statistical baselines define the accuracy–latency/energy Pareto frontier in regions of high domain shift and low forecastability. | Non-dominated sorting over mean MASE vs. latency (`ms/pt`) and TDP-estimated CPU energy (`mWh/1k points`), mapping feature-space decision boundaries where marginal cost per 1% MASE gain becomes prohibitive. |

---

## 2. Datasets and Pretraining Overlap Registry

### 2.1 Benchmark Corpus (GIFT-Eval Leaderboard)

The primary evaluation corpus is Salesforce `GIFT-Eval` (Aksu et al., 2024; arXiv:2410.10393; snapshot date `2026-10-16`), encompassing 97 benchmark configurations across 28 distinct `base_dataset` clusters and 7 domain categories (`Energy`, `Transport`, `Nature`, `Web/CloudOps`, `Sales`, `Econ/Fin`, `Healthcare`).

- **Leaderboard Source:** <https://github.com/SalesforceAIResearch/gift-eval>
- **Hugging Face Data:** <https://huggingface.co/datasets/Salesforce/GiftEval>
- **Pretraining Pool:** <https://huggingface.co/datasets/Salesforce/GiftEvalPretrain>

**Local Subsampled Benchmark Stratum (11 configurations):**
For local execution, evaluation follows GIFT-Eval's train/test split logic (`TEST_SPLIT = 0.1`, rolling test windows), subsampled to 60 series per dataset and 2 evaluation windows per series (`configs/experiment.yaml`). Series subsampling is deterministically seeded via `np.random.default_rng(42).choice(total_rows, size=60, replace=False)`.

| Configuration | Domain | Freq | Horizon $H$ | Season Length $m$ (MASE) | STL Period | Known Test-Data Overlap (`dataset_leak`) | Licence / Data Source |
|---|---|---|---|---|---|---|---|
| `m4_monthly/M/short` | Econ/Fin | Monthly | 18 | 12 | 12 | Yes (`Chronos_small`, `Chronos_base`, `Chronos_large` pretraining pool) | Open (M4 Competition / Monash Archive) |
| `m4_quarterly/Q/short` | Econ/Fin | Quarterly | 8 | 4 | 4 | Yes (`Chronos` pretraining pool) | Open (M4 Competition / Monash Archive) |
| `ett1/D/short` | Energy | Daily | 30 | 1 | 7 | None documented | Apache-2.0 (ETDataset, Zhou et al., 2021) |
| `solar/W/short` | Energy | Weekly | 8 | 1 | 52 | None documented | Public Domain (NREL Solar Power Data) |
| `hospital/M/short` | Healthcare | Monthly | 12 | 12 | 12 | None documented | Open (expsmooth / Monash Archive) |
| `us_births/M/short` | Healthcare | Monthly | 12 | 12 | 12 | None documented | Public Domain (CDC National Vital Statistics) |
| `saugeen/W/short` | Nature | Weekly | 8 | 1 | 52 | None documented | Open (Environment and Climate Change Canada) |
| `jena_weather/D/short` | Nature | Daily | 30 | 1 | 7 | None documented | CC-BY-4.0 (Max Planck Institute for Biogeochemistry) |
| `hierarchical_sales/W/short` | Sales | Weekly | 8 | 1 | 52 | None documented | Open (Monash Retail Sales Archive) |
| `loop_seattle/D/short` | Transport | Daily | 30 | 1 | 7 | None documented | Open (Washington State Department of Transportation) |
| `bizitobs_l2c/H/short` | Web/CloudOps | Hourly | 48 | 24 | 24 | None documented | Open (BizITObs Enterprise Telemetry) |

*Note on Season Length vs. STL Period:* In accordance with GluonTS / GIFT-Eval conventions, weekly and daily series use `season_length = 1` for MASE scale computation, whereas time-series feature extraction (STL decomposition) uses the natural physical calendar cycle (`STL period = 52` for weekly, `7` for daily, `12` for monthly, `24` for hourly).

### 2.2 Curated Evaluation Datasets (Not in Any Documented Pretraining Corpus)

To test generalization under controlled domain shift and temporal recency, two datasets were curated and processed without pretraining exposure:

| Dataset Configuration | Evaluation Stratum | Description & Processing Rules | Sample Size & Horizon | Licence & Source |
|---|---|---|---|---|
| `unseen_mandi/W/short` | `unseen_dataset_overlapping_domain` (Econ/Fin) | Weekly wholesale agricultural commodity prices (Wheat, Maize, Soyabean, Lentil, Green Gram, Arhar) across Indian mandis (2012–2016). Missing-week filter: $\ge 75\%$ of 240 calendar weeks ($\ge 180$ observed records), nearest-Monday date snapping within $\pm 4$ days, linear interpolation + forward/backward fill. | 60 series, 261 weeks each (15,660 observations). Horizon $H = 8$, MASE $m = 1$, STL period 52. | Government Open Data License (GODL-India); `data.gov.in` / Agmarknet, archived via Covert (2020). |
| `unseen_recency/D/short` | `recency` (Post-Cutoff Meteorological) | Daily meteorological observations across 20 global metropolitan cities (`Tokyo`, `Delhi`, `Shanghai`, `Sao_Paulo`, `Mexico_City`, `Cairo`, `Mumbai`, `Beijing`, `Dhaka`, `Osaka`, `New_York`, `Karachi`, `Buenos_Aires`, `Chongqing`, `Istanbul`, `Kolkata`, `Manila`, `Lagos`, `Rio_de_Janeiro`, `London`) tracking 3 variables: `temperature_2m_mean`, `shortwave_radiation_sum`, `wind_speed_10m_max` from 2025-01-01 to 2026-09-30. Strictly post-dates pretraining cutoffs of `Chronos_small` (2024-03) and `Moirai_small` (2024-06). 2025+ checkpoints (`TimesFM-2.5`, `Chronos-2`, `Moirai2`) are coded as `"unknown"` in the recency registry. | 60 series, 638 days each (38,280 observations). Horizon $H = 30$, MASE $m = 1$, STL period 7. | CC-BY-4.0; ECMWF ERA5 reanalysis via Open-Meteo Archive API. |
| S&P 500 Daily OHLC | Financial Volatility Case Study (H3) | Daily open, high, low, close prices from 1990-01-02 to 2026-10-15 (9,247 trading days) from a single frozen data file (`data/raw/spx_daily.csv`, Yahoo Finance `^GSPC`). Garman–Klass volatility proxy: $\hat{\sigma}^2_{\text{GK},t} = 0.5[\log(H_t/L_t)]^2 - (2\log 2 - 1)[\log(C_t/O_t)]^2$, bounded below by $10^{-8}$. | 9,247 trading days; rolling 512-day training context, evaluated at horizons $H \in \{1, 5, 20\}$. | Single frozen data file; Yahoo Finance `^GSPC`. |

*Community Checkpoint Exclusion:* Community fine-tunes such as `Viharikvs/TFSMAG` were evaluated and excluded due to unverified training weights, absence of reproducible pretraining metadata, and manual gating restrictions.

*Five Target Datasets for Future Out-of-Domain Stratum Expansion:*
To expand out-of-domain evaluation beyond agricultural commodity prices into domains entirely absent from all documented TSFM corpora, five non-overlapping domain datasets are formally specified:
1. **Physiological / Neurological Telemetry:** CHB-MIT Scalp EEG Database (PhysioNet; Goldberger et al., 2000; OOD Domain: `Neurology/Biomedical`).
2. **Astrophysical Photometry:** Kepler / TESS Exoplanet Transit Light Curves (NASA Exoplanet Archive; Borucki et al., 2010; OOD Domain: `Astrophysics`).
3. **Marine Acoustic Propagation:** Ocean Observatories Initiative (OOI) Hydrophone Array Acoustic Pressure Series (NSF / Rutgers; OOD Domain: `Oceanography/Acoustics`).
4. **Geophysical Seismology:** Global Seismographic Network (GSN) Broadband Ground Motion Velocity (IRIS / USGS; OOD Domain: `Seismology`).
5. **Wildlife Bioacoustic Monitoring:** Great Tit (*Parus major*) Territorial Vocalization Pitch & Energy Contours (Bioacoustics Research Program; OOD Domain: `Ecology/Bioacoustics`).

In accordance with Requirement A.4, any domain stratum with fewer than 5 base datasets (`unseen_dataset_overlapping_domain` with 1 base dataset, `recency` with 1 base dataset) is reported **descriptively only** (`inference_status = "descriptive_only (<5 base datasets)"`).

### 2.3 Time-Series Meta-Features (Predictors)

All meta-features are extracted strictly from the z-scored training contexts (before evaluation windows) and aggregated to dataset configuration level by median:

| Descriptor | Mathematical Definition | Computational Specification & Edge-Case Handling | Source Reference |
|---|---|---|---|
| `forecastability` | $1 - H_{\text{spec}}$, where $H_{\text{spec}} = -\int_{-\pi}^{\pi} f(\lambda)\log f(\lambda)d\lambda / \log(\pi)$ | Normalized spectral entropy via Welch periodogram. Entered in regressions; `spectral_entropy` is omitted to avoid exact collinearity. | Goerg (2013) |
| `seasonal_strength` | $\max(0, 1 - \frac{\mathrm{Var}(R)}{\mathrm{Var}(S + R)})$ | From additive STL decomposition. When $T < 2 \times \text{period} + 1$, falls back to OLS linear detrending. | Hyndman & Athanasopoulos (2021) |
| `trend_strength` | $\max(0, 1 - \frac{\mathrm{Var}(R)}{\mathrm{Var}(T + R)})$ | From additive STL decomposition with robust loess smoothing. | Hyndman & Athanasopoulos (2021) |
| `hurst` | Detrended Fluctuation Analysis (DFA) scaling exponent $\alpha$ | Fluctuation function $F(s) \propto s^\alpha$ over dyadic scales $s \in [4, \lfloor T/4 \rfloor]$ using polynomial detrending (replaces heuristic R/S). | Peng et al. (1994) |
| `stability` | $\mathrm{Var}(\bar{y}_w) / \mathrm{Var}(y)$ | Variance of non-overlapping window means with tile width $w = \max(\text{stl\_period}, 10)$. | Hyndman et al. (2023, `tsfeatures`) |
| `lumpiness` | $\mathrm{Var}(s^2_w) / \mathrm{Var}(y)^2$ | Variance of non-overlapping window variances with tile width $w = \max(\text{stl\_period}, 10)$. | Hyndman et al. (2023, `tsfeatures`) |
| `log_length` | $\log(T)$ | Natural logarithm of the series training context length. | Standard formulation |

All 7 meta-features demonstrate low collinearity ($\mathrm{VIF} < 3.5$; full diagnostics in `results/features/vif_report.csv` and `results/features/vif_full_predictors.csv`).

---

## 3. Models and Unified Classical Baseline Set

### 3.1 Time-Series Foundation Models (15 Models Total)

**Locally Executed Checkpoints (4 models, context length 512, z-score scaled):**
1. **Chronos-T5-small** (46M parameters): Hugging Face `amazon/chronos-t5-small` (Ansari et al., 2024; arXiv:2403.07815).
2. **Moirai-1.1-R-small** (14M parameters): Hugging Face `Salesforce/moirai-1.1-R-small` (Woo et al., 2024; arXiv:2402.02592).
3. **TimesFM-2.5-200m** (231M parameters): Hugging Face `google/timesfm-2.5-200m-pytorch` (Das et al., 2024; arXiv:2310.10688).
4. **TTM-R2-Pretrained** (0.8M parameters): Hugging Face `ibm-granite/granite-timeseries-ttm-r2` (Ekambaram et al., 2024; arXiv:2401.03955).

*Context-Length Handling for Short Series:* For input series shorter than the 512 context length, `Moirai` applies left-padding with zeros and sets the boolean mask `past_is_pad = True`. `Chronos`, `TimesFM`, and `TTM` receive the series at native sequence length without artificial truncation.

**Leaderboard-Only Checkpoints (11 models):**
Ingested directly from the official GIFT-Eval benchmark table:
`Chronos_base`, `Chronos_large`, `chronos_bolt_small`, `chronos_bolt_base`, `Chronos-2`, `Moirai_base`, `Moirai_large`, `Moirai2`, `TimesFM` (1.0-200m), `timesfm_2_0_500m`, and `TTM-R1-Pretrained`.

### 3.2 Unified Classical Baseline Set

To ensure strict comparability across leaderboard and local configurations (as the GIFT-Eval leaderboard excludes `Drift` and `LightGBM`), the `"best classical"` benchmark is defined identically for all 99 configurations as the lowest MASE among the **common 5-model classical set**:
1. `Naive` (random walk persistence)
2. `SeasonalNaive` (seasonal repetition at GluonTS period $m$)
3. `AutoETS` (Hyndman automated exponential smoothing via StatsForecast)
4. `AutoTheta` (assimilation of linear trend with seasonal decomposition)
5. `AutoARIMA` (Hyndman–Khandakar stepwise AIC model selection)

*LightGBM Tabular Baseline:* Tuned via Optuna under a strict wall-clock budget matched to `Moirai_small` CPU inference; evaluated and reported as a separate machine-learning baseline in Phase 8 without contaminating the classical reference.

---

## 4. Methodological Workflow by Phase (Phases 0–9)

### Phase 0: Test Harness Verification & Exact Anchor Replication
- **Verification Anchor:** `AirPassengers` monthly passenger series (1949–1960) serves as the ground-truth smoke test for point and interval metrics.
- **6-Decimal Benchmark Replication:** Local `Seasonal_Naive` MASE implementation was validated against the published GIFT-Eval leaderboard across four uncapped configurations (`hospital/M`, `us_births/M`, `saugeen/W`, `solar/W`) with `exact_gift_eval = True` (full series, no subsampling), reproducing leaderboard numbers to 6 decimal places (`abs_diff = 0.000000`).

### Phase 1: Pretraining Exposure Registry & Leakage Tracking
- Formulates `configs/pretraining_registry.yaml` mapping 15 models $\times$ 7 domains.
- Models pretraining domain coverage as `true`, `false`, `unknown`, or `partial`. Specifically, `Chronos` is marked as `partial` on `Econ/Fin` (`"no market-price data; finance via M4 only"`, Ansari et al., 2024 Table 1).
- Encodes known dataset leakage as a binary flag `dataset_leak` (e.g., `m4_monthly` and `m4_quarterly` in Chronos's pretraining pool).

### Phase 2: Classical Baselines & Unified Reference Computation
- Executes the 5 common classical models across all 11 benchmark configurations and 2 unseen configurations.
- Computes `best_classical_local.parquet` and `best_classical_leaderboard.parquet` using the identical 5-model candidate pool.

### Phase 3: Meta-Feature Extraction & Domain Similarity Reference Pools
- Computes 7 series meta-features per series, aggregating by median to configuration level (99 configs).
- Defines the leave-one-base-dataset-out pretraining reference pool $\mathcal{R}_{m,-b(d)}$ for model $m$ and test dataset $d$:
  $$\mathcal{R}_{m,-b(d)} = \{d' \in \mathcal{D}_{\text{pretrain}}(m) : \text{base\_dataset}(d') \ne \text{base\_dataset}(d)\}$$
- Computes regularized Mahalanobis distance `sim_distance` with ridge regularization ($0.25 \mathbf{I}$) and $k$-NN distance ($k=3$).

### Phase 4: Zero-Shot Foundation Model Inference & Compute Profiling
- Profiles local zero-shot inference across 13 configurations over 3 warmup-stabilized repeats.
- Primary hardware metric: per-point latency (`infer_time_ms_per_point`, $\text{ms/pt}$), reported with median and IQR.
- CPU energy consumption is estimated deterministically based on host CPU TDP (`45 W`) via CodeCarbon (`energy_source = "TDP_estimate"`). Dedicated VRAM metrics are omitted on CPU hardware.

### Phase 5: Cross-Series Per-Horizon Conformal Calibration
- Evaluates native 80% prediction intervals ($q_{0.1}$ to $q_{0.9}$) against split conformal calibration.
- **Calibration Protocol:** Calibrates on rolling window 1 across all 60 series and evaluates on rolling window 2, executed **per horizon step** $h \in \{1, \dots, H\}$.
- **Exchangeability Assumption:** Assumes exchangeability of standardized nonconformity scores across cross-sectional series within the same dataset configuration at matching forecast horizon steps.

### Phase 6: Hierarchical Mixed-Effects Models & LODO Cross-Validation
- **Response Variable:** Modeled as the log-ratio:
  $$y_{m,d} = \log\left(\frac{\mathrm{MASE}_{m,d}}{\mathrm{MASE}_{\text{classical},d}}\right)$$
  Reported with $\Delta\mathrm{MASE} = -y_{m,d}$ (so positive values indicate TSFM superiority) and percentage gain:
  $$\%\text{ Gain} = (1 - e^y) \times 100\%$$
- **Model Ladder (Fitted via ML for LRT/AIC/BIC, REML for Coefficients):**
  - `M0`: Crossed random intercepts $(1 \mid \text{base\_dataset}) + (1 \mid \text{config}) + (1 \mid \text{model\_family})$.
  - `M1`: `M0` + 7 series descriptors + $\log(\text{params}) + \text{dataset\_leak} + \text{source\_local} + \text{frequency\_group}$.
  - `M2a`: `M1` + categorical similarity `sim_binary`.
  - `M2b`: `M1` + continuous Mahalanobis distance `sim_distance`.
  - `M2`: `M1` + full similarity block (`sim_binary`, `sim_share`, `sim_distance`).
  - `M3`: `M2` + domain shift $\times$ difficulty interactions (`sim_distance:forecastability`, `sim_distance:seasonal_strength`, `sim_distance:log_length`).
- **Inference Sample:** 1,463 TSFM-only observations (97 GIFT-Eval configs $\times$ 15 TSFMs + 2 unseen configs $\times$ 4 local TSFMs). `source_local` controls for pooled local execution.
- **Multiple Testing:** Benjamini–Hochberg False Discovery Rate (BH-FDR) $q$-values are computed over the pre-specified family of similarity fixed effects (3 terms in `M2`; 6 similarity + interaction terms in `M3`).
- **LODO Out-of-Sample CV:** Evaluates out-of-sample ranking ($\rho$) and sign accuracy across 30 held-out `base_dataset` clusters against the Tan Jerome & Simon (2026; arXiv:2602.05147) two-feature heuristic ($n_{\text{train}} < 700$ and $\text{seasonal\_strength} \ge 0.30$).

### Phase 7: Financial Volatility Structural-Break Case Study (Testing H3)
- Evaluates S&P 500 daily Garman–Klass log-volatility across 9,247 trading days (1990–2026).
- **Breakpoint Detection:** PELT algorithm (Killick et al., 2012) on 5-day smoothed standardized log-volatility with penalty $\text{pen} = 12.0$ identifies 30 major structural breaks, recovering 4 of 6 labelled crisis dates within $\pm 30$ trading days.
- **Regime Evaluation:** Forecasts evaluated across 24 post-break origins (within 63 trading days of break) and 24 stable origins ($>90$ trading days from any break) at horizons $H \in \{1, 5, 20\}$.
- **Loss Functions & Statistical Tests:** Evaluated on QLIKE, MSE, and MASE. Two-sided Diebold–Mariano (1995) tests with Harvey–Leybourne–Newbold adjustment compare `Chronos_small` vs. `Moirai_small`. Interval exceedances ($[q_{0.1}, q_{0.9}]$, nominal coverage 80%, $\alpha = 0.20$) are assessed via Kupiec (1995) Proportion of Failures (POF) $\chi^2(1)$ tests.

### Phase 8: Robustness Ablations & Tabular ML Parity
- **Ablation 8a:** LightGBM baseline tuned under Moirai-small wall-clock budget parity.
- **Ablation 8b.1:** Common z-score scaling vs. model-default internal scaling across 4 configurations.
- **Ablation 8b.2:** Refitting mixed models after strictly excluding all pretraining overlap (`dataset_leak == 1`).
- **Ablation 8b.3:** Sensitivity of `M2` to coding unverified registry entries as conservative (`unknown -> False`) vs. optimistic (`unknown -> True`).
- **Ablation 8b.4:** Bayesian Online Changepoint Detection (BOCPD; Adams & MacKay, 2007) vs. PELT breakpoint sensitivity.
- **Ablation 8b.5:** Gamma log-link Generalized Linear Model (GLM) fit directly on raw ratio $\mathrm{MASE}_{\text{model}} / \mathrm{MASE}_{\text{classical}}$ with cluster-robust standard errors.

### Phase 9: Compute-Efficiency Pareto Frontier & Programmatic Reporting
- Evaluates multi-objective non-dominated frontiers across accuracy (MASE), latency ($\text{ms/pt}$), and energy ($\text{mWh/1k points}$).
- Computes marginal compute cost per 1% MASE improvement strictly over configurations with positive gain ($\Delta\mathrm{MASE} > 0$).
- Maps feature-space decision boundaries across domain shift, forecastability, and sequence length regimes.
- Programmatically generates all publication figures and `report/report.md`.

---

## 5. Key Empirical Findings (Traced from `results/`)

- **H1 Confirmed (Similarity Predicts Beyond Characteristics):** In the mixed-effects ladder, adding pretraining similarity terms significantly improves fit over the series characteristics baseline `M1` (LRT $\chi^2(3) = 14.13$, $p = 0.0027$; $\Delta\mathrm{AIC} = -8.1$). Adding shift $\times$ feature interactions (`M3`) further improves fit ($\chi^2(3) = 13.82$, $p = 0.0032$; $\Delta\mathrm{AIC} = -7.8$). Comparing single-feature additions shows continuous Mahalanobis distance `M2b` outperforms binary coverage `M2a` ($\Delta\mathrm{AIC} = -4.62$, $\Delta\mathrm{BIC} = -4.62$).
- **H2 Confirmed (Out-of-Domain Degradation):** TSFMs outperform the best classical baseline on 66.0% of in-domain pairs and 67.7% of partial-overlap pairs, but win rate drops to **31.5%** on out-of-domain datasets (mean relative degradation of $-17.7\%$, $t = -3.72$, $p = 0.0025$). Strata with $< 5$ base datasets are documented descriptively only.
- **LODO Out-of-Sample Ranking:** The Tan Jerome & Simon (2026) heuristic achieves near-zero rank correlation with empirical zero-shot gains ($\rho = 0.001$, sign accuracy 53.0%). Incorporating continuous domain distance (`M2b`) raises rank correlation to $\rho = 0.174$ ($p = 2.1 \times 10^{-11}$) with 62.4% sign accuracy.
- **H3 Structural-Break Volatility Contrast:** In stable market windows, `Moirai_small` significantly outperforms `Chronos_small` at horizon $H = 1$ (mean difference $0.132$ MASE, 95% bootstrap CI $[0.017, 0.262]$) and $H = 5$ ($0.066$ MASE, CI $[0.025, 0.107]$). Post-break, neither foundation model separates from rolling `GARCH(1,1)-t`, and the Difference-in-Differences contrast confirms convergence in crisis regimes.
- **H4 Compute-Efficiency Frontier:** `SeasonalNaive`, `AutoTheta`, `AutoETS`, `TTM-R2-Pretrained` ($0.8\text{M}$ params, median latency $0.05\text{ ms/pt}$), `Moirai_small` (median latency $0.56\text{ ms/pt}$), and `TimesFM-2.5` form non-dominated segments of the Pareto frontier. In high-shift, low-forecastability regimes, classical models dominate zero-shot TSFMs on both accuracy and compute efficiency.

---

## 6. Methodological Scope and Limitations

1. **Local vs. Published Leaderboard Execution:** Local inference was executed across 13 configurations for 4 TSFMs and 6 classical models. Cross-model regressions across 15 TSFMs incorporate published GIFT-Eval leaderboard numbers, introducing minor cross-environment implementation variance controlled via the `source_local` fixed effect.
2. **Pretraining Corpus Transparency:** Commercial and semi-open foundation models disclose domain categories rather than row-level dataset inventories. The pretraining registry explicitly marks unverified claims, and Phase 8 verifies that statistical conclusions are invariant under conservative vs. optimistic coding.
3. **Hardware Environment:** Local inference was profiled on host CPU with deterministic TDP energy modeling ($45\text{ W}$). Latency ($\text{ms/pt}$) is reported as the primary hardware-agnostic benchmark, supplemented with median and IQR dispersion.

---

## 7. Complete Reference List

- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. *arXiv:0710.3742*.
- Aksu, T., et al. (2024). GIFT-Eval: A Benchmark for General Time Series Forecasting Foundation Models. *arXiv:2410.10393*.
- Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. *arXiv:2403.07815*.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Borucki, W. J., et al. (2010). Kepler Planet-Detection Mission: Introduction and First Results. *Science*, 327(5968), 977–980.
- Covert, I. (2020). Agmarknet Agricultural Commodity Price Dataset. GitHub repository: `iancovert/Agmarknet`.
- Das, A., et al. (2024). A decoder-only foundation model for time-series forecasting (TimesFM). *arXiv:2310.10688*.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Ekambaram, V., et al. (2024). Tiny Time Mixers (TTMs): Fast Pre-trained Models for Extreme Minimal Transfer Learning of Multivariate Time Series. *arXiv:2401.03955*.
- Garman, M. B., & Klass, M. J. (1980). On the Estimation of Security Price Volatilities from Historical Data. *Journal of Business*, 53(1), 67–78.
- Goerg, G. M. (2013). Forecastable Component Analysis. *International Conference on Machine Learning (ICML)*, 64–72.
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation*, 101(23), e215–e220.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts: Melbourne, Australia.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Hyndman, R. J., Kang, Y., Montero-Manso, P., Talagala, T., Wang, E., Yang, Y., & O'Hara-Wild, M. (2023). `tsfeatures`: Time Series Feature Extraction. R package version 1.1.1.
- J.P. Morgan / Reuters. (1996). *RiskMetrics Technical Document* (4th ed.). New York.
- Karaouli, M., et al. (2025). How Foundational are Foundation Models for Time Series Forecasting? *arXiv:2502.04107*.
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal Detection of Changepoints With a Linear Computational Cost (PELT). *Journal of the American Statistical Association*, 107(500), 1590–1598.
- Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *The Journal of Derivatives*, 3(2), 73–84.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., & Goldberger, A. L. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*, 49(2), 1685–1689.
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
- Tan Jerome, A., & Simon, N. (2026). When Do Time Series Foundation Models Beat Classical Methods? A Break-Even Analysis. *arXiv:2602.05147*.
- Woo, G., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers (Moirai). *arXiv:2402.02592*.
- Xue, W., et al. (2026). QuitoBench: Benchmarking Time Series Foundation Models across Diverse Dataset Characteristics. *arXiv:2602.07188*.
- Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (ETDataset). *AAAI Conference on Artificial Intelligence*, 35(12), 11106–11115.
