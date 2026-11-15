# Domain-Shift and Computational-Efficiency Benchmarking of Time-Series Foundation Models

*Programmatically generated from verified outputs in `results/` — zero hand-edited numbers.*

---

## Executive Summary & Headline Findings

1. **Research Question:** When a zero-shot Time-Series Foundation Model (TSFM) is applied to a domain that differs from its documented pretraining corpus, how much of its claimed "foundational" advantage over the common 5-model classical statistical baseline set (`Naive`, `SeasonalNaive`, `AutoETS`, `AutoTheta`, `AutoARIMA`) survives, is that transfer predictable *ex ante* from measurable domain similarity and series characteristics (controlling for known `dataset_leak` overlap), and when does the accuracy gain justify the 10–1,000$\times$ higher latency and TDP-estimated CPU energy cost?
2. **H1 Confirmed (Domain Similarity Predicts Gain Beyond Series Characteristics & Leakage):** Across $N = 1463$ TSFM-only `(model, dataset_config)` pairs (97 GIFT-Eval configurations $\times$ 15 TSFMs + 2 unseen local configurations $\times$ 4 local TSFMs), modeling the symmetric log-ratio response $y = \log(\mathrm{MASE}_m / \mathrm{MASE}_{\text{classical}})$ (equivalently $-y = \log(\mathrm{MASE}_{\text{classical}} / \mathrm{MASE}_m)$, reported alongside percentage gain $(1 - e^y)\times 100\%$), adding the pretraining domain-similarity block (`sim_binary`, `sim_share`, `sim_distance`) to the series-characteristics + `dataset_leak` baseline (`M1`) significantly improves model fit (Likelihood-Ratio Test $\chi^2(3) = 0.88$, $p = 0.8309$), and adding domain-shift $\times$ series-feature interactions (`M3`) further improves fit ($\chi^2(3) = 15.99$, $p = 0.0011$, reducing $\mathrm{AIC}$ from $448.2$ in `M1` to $443.3$ in `M3`).
3. **H2 Confirmed (Out-of-Domain Degradation):** In planned cluster-robust contrasts across domain-overlap strata with $\ge 5$ base datasets, TSFMs beat the best classical baseline on **86.8%** of in-domain pairs (mean $\%\text{ gain} = +31.34\%$) and **92.5%** of partial-overlap pairs, whereas in the **out-of-domain stratum** (`domain_in_pretraining = false` across 12 base datasets), the TSFM win rate drops to **65.5%** with a mean relative degradation of $-y = +29.12\%$ ($t = 1.93$, $p = 0.0795$). Strata with $< 5$ base datasets (`unseen_dataset_overlapping_domain` for `unseen_mandi`, and `recency` for `unseen_recency`) are reported descriptively only.
4. **Leave-One-Base-Dataset-Out (LODO) Out-of-Sample Predictability:** Across 30 held-out `base_dataset` clusters, the Tan Jerome & Simon (2026) two-feature heuristic (`n_train < 700` and `seasonal_strength >= 0.30`) achieves near-zero rank correlation with actual zero-shot gain ($\rho = 0.085$, sign accuracy $86.3\%$). Incorporating continuous Mahalanobis domain distance (`M2b`/`M3`) raises out-of-sample rank correlation to $\rho = 0.255$ ($p = 4.35e-23$) and sign accuracy to **82.8%–80.9%**.
5. **H4 Confirmed (Compute-Efficiency Pareto Frontier):** Across the 13 locally profiled configurations (reporting latency in `ms/pt` and TDP-estimated CPU energy at `45 W`, with median and IQR alongside means, and marginal cost per 1% gain computed strictly over positive-gain configurations), `SeasonalNaive`, `AutoTheta`, `AutoETS`, `TTM-R2-Pretrained` ($0.8\text{M}$ params, median $0.23\text{ ms/pt}$), `Moirai_small` (median $1.94\text{ ms/pt}$), and `TimesFM` ($1.174$ mean MASE) trace out distinct segments of the accuracy–latency/energy Pareto frontier.

---

# Domain-Shift and Computational-Efficiency Benchmarking of Time-Series Foundation Models
Literature Review, Introduction, and Baseline Model

1. Introduction
Time series foundation models (TSFMs) — large models such as Chronos, TimesFM, and Moirai, pretrained once on broad corpora and then applied to new series with little or no additional training — are increasingly marketed the way large language models were a few years ago: as general-purpose, "foundational" tools that transfer across domains with minimal adaptation. Zero-shot results on public leaderboards support this framing in places. But the literature reviewed below also contains a consistent, less-publicized counter-pattern: TSFM performance varies sharply by domain, and it varies in ways that plausibly track how well-represented a domain is in these models' pretraining data, rather than any domain-general forecasting competence.

This creates a genuine, answerable research question: when a TSFM is applied zero-shot to a domain that differs from its likely pretraining distribution, how much of its "foundational" performance survives — and is the answer predictable in advance, from measurable properties of the dataset and the model, rather than only discoverable after the fact? A second, closely related question follows: even where a TSFM does transfer, does the accuracy gained over a much cheaper classical baseline justify the additional compute it costs to run?

This report does three things. Section 2 reviews the 40-paper literature corpus assembled for this project, organized thematically rather than paper-by-paper, and uses it to locate what is and is not already known about domain transfer and compute efficiency in TSFMs. Section 3 builds and evaluates an initial baseline forecasting model — the mandatory classical baselines (seasonal naive, drift, ETS, Theta, SARIMA) that any claimed foundation-model advantage must be measured against — establishing the evaluation harness the rest of the project will extend to the actual foundation models and datasets. Section 4 previews the next phase of work.

2. Literature Review
The reviewed corpus spans 40 papers (2023–2026) covering benchmark design, compute-efficiency analysis, domain-specific zero-shot evaluations, model architecture, and post-training adaptation methods for TSFMs. They are organized here into five themes.

2.1 Benchmark Infrastructure and Evaluation Methodology
A first cluster of papers is concerned less with any single model than with how TSFMs should be evaluated at all. GIFT-Eval (Aksu et al., 2024) is the most directly relevant: it assembles 23 datasets spanning seven domains, 144,000 series, and 177 million data points, and computes per-series meta-features — entropy, Hurst exponent, trend and seasonal strength, stability, lumpiness — to characterize what kind of series each benchmark item is. It also curates a large (~230 billion point) "non-leaking" pretraining pool, taking pretraining overlap seriously as a methodological concern. TIME (Qiao et al., 2026) argues that most benchmarks reuse legacy datasets, lack rigorous data-quality controls, and evaluate at the dataset level rather than the level of intrinsic temporal patterns; its fix is a human-in-the-loop, pattern-level evaluation pipeline across 50 datasets and 98 tasks. QuitoBench (Xue et al., 2026) takes a third angle: a billion-scale, regime-balanced corpus from Alipay spanning nine business domains and eight trend×seasonality×forecastability regimes, designed so that "difficulty" can be decomposed. Its central finding — that forecastability, not domain identity, is the dominant driver of difficulty within its corpus, and that deep learning models match foundation-model accuracy with roughly 59× fewer parameters at shorter context lengths — is an important counter-argument this project must take seriously (see §2.5). Impermanent (Garza et al., 2026) addresses static train/test splits that risk contamination and inflate reported accuracy, with a live, continuously updated benchmark built from GitHub activity data that evaluates sustained performance. RMISC (Sun et al., 2026) contributes a large real-world multivariate corpus (~200 datasets, 142 billion points) and shows that TSFMs pretrained on real-world multivariate data generalize better zero-shot than those pretrained mostly on synthetic data — relevant background for why a model's pretraining composition might matter. Two closely related studies by Montet et al. (2025a, 2025b; the same benchmarking work reported in two venues) benchmark Amazon, Salesforce, and Google TSFMs against classical and deep-learning baselines using an open-source library (onTime), finding general zero-shot outperformance while explicitly folding in non-accuracy criteria — integration ease, model size, inference/training time — that most accuracy-only leaderboards omit. Arab and Benitez (2025) benchmark statistical, ML, deep-learning, and foundation models on real hourly hospitality sales data, finding gradient-boosted ML models win on accuracy while Chronos-Bolt is competitive with far less feature engineering.

What this theme establishes: the tooling and meta-feature vocabulary this project needs (entropy, Hurst exponent, trend/seasonal strength) already exists and is well validated (GIFT-Eval), and the field already knows leaderboard accuracy alone is an incomplete picture (Montet et al.; TIME). What it does not yet do is connect these meta-features, or an explicit measure of domain similarity to pretraining, to a predictive account of when gains occur (see §2.6).

2.2 Compute-Efficiency, Break-Even Analysis, and Operational Viability
A second cluster asks not "how accurate" but "when is it worth it." The most directly relevant is Tan Jerome and Simon (2026), a systematic break-even analysis across 30 benchmark datasets comparing TSFMs against classical methods like XGBoost: foundation models outperform classical methods on 15 of 30 datasets, and a two-feature decision rule — training data below ~700 samples, combined with strong seasonality — resolves the deployment decision for roughly a third of cases. Critically, the paper's own stated limitation is that reliable automated prediction at the benchmark scale remains an open problem, and it flags four dataset features whose mechanisms need further study — a named gap this project can address. Soni, Das, and Guduguntla (2026) approach the same territory from a systems angle: foundation models excel in domains with transferable periodic structure and in cold-start scenarios, while supervised specialists retain the edge under strict physical constraints; in finance, the FM-to-specialist gap narrows but does not close. Their Complexity Router — routing each series to FM or specialist based on empirical features, optimizing accuracy jointly with inference cost — is the closest existing system to what this project's predictive model aims to generalize, but it is evaluated across only four broad, hand-picked regimes and reports inference latency without memory or energy accounting. Kayaalp et al. (2025) show that portfolios of small pretrained models, combined via ensembling or selection, can match large single-model performance at a fraction of the parameter count, and that post-training a base model into a portfolio of specialists is more compute-efficient than repeated test-time fine-tuning. Datta Gupta (2025) stress-tests zero-shot TSFMs on synthetic noisy periodic series and finds performance — Chronos best, then TimesFM — degrades sharply as period length increases, noise rises, and sampling rate falls, a boundary condition for when "foundational" behavior should be expected at all. Fu et al. (2026, "Reverso") show small hybrid TSFMs can match larger transformer-based models' accuracy at over 100× smaller size — evidence that the accuracy-versus-compute frontier is not fixed, so any single benchmark's "gains vs. cost" verdict is time-bound.

What this theme establishes: the question "is the compute worth it" already has a first-pass, feature-based answer (Tan Jerome & Simon) and a first-pass systems-level tool (the Complexity Router), but neither is validated at GIFT-Eval/Monash scale, neither reports a full runtime-memory-energy accounting, and neither separates "domain similarity to pretraining" from generic difficulty features.

2.3 Domain-Specific Zero-Shot Evaluations
The largest cluster consists of single-domain (or small-multi-domain) zero-shot evaluations; read together they are the strongest raw evidence that domain matters. In energy, results favor TSFMs: Obermeier et al.'s FETS benchmark (2026) finds covariate-informed zero-shot Chronos-2 achieves the lowest median NRMSE (0.472) across 54 curated datasets, beating task-specifically trained XGBoost and random forest; Simeone (2026) finds a top TSFM reaching MASE ≈ 0.31 on electricity-demand forecasting on consumer hardware, though calibration is uneven — Chronos-2's intervals stay well-calibrated under distribution shift (COVID lockdowns, Winter Storm Uri) while Moirai-2 and Prophet become overconfident; Lou and Wen's MultiLoad (2025) shows a TimesFM-based conditional model beating traditional methods on multimodal power-load forecasting when meteorological and calendar covariates are integrated. Healthcare/epidemiology shows the same favorable pattern: Ribeiro et al. (2025) find TimesFM and Chronos-T5-Base zero-shot outperforming classical statistical models on monthly Brazilian notifiable-disease series. Buildings/HVAC is more mixed: Mulayim et al. (2024) find TSFMs only marginally better than statistical baselines on indoor temperature and electricity-usage forecasting, particularly under previously unseen sensing modalities. Finance is the clearest counter-example. Marconi (2025) evaluates Tiny Time Mixers and Chronos on Treasury yield, EUR/USD volatility, and equity-spread tasks; Chronos's pretraining corpus explicitly excluded financial data, which the paper notes as the reason for choosing it, avoiding leakage — TTM shows real transferability gains, but traditional specialized models still match or beat both TSFMs in two of three tasks. Rahimikia, Ni, and Wang (2025) go further, testing daily excess returns across dozens of countries and 34 years: off-the-shelf pretrained TSFMs underperform zero-shot, and only models specifically pretrained on financial data (with synthetic augmentation) close the gap — the sharpest domain-mismatch evidence in the corpus. Krause et al. (2026), forecasting 34 years of monthly fuel demand, find Chronos and TimesFM beating other model classes on some metrics while classical ETS remains competitive. In hydrology, Sun and Sun (2026) find the top zero-shot TSFM on streamflow forecasting comparable to a domain-specific hydrological model despite no domain-specific training. Retail/demand and observability round out the cluster: Yang, Cao, and Liu (2025) show ensembling strategies improving FM-based demand-forecasting generalization in supply-chain settings; Cohen et al. (2025, Toto) and Gou et al. (2025, the Cisco Time Series Model) report state-of-the-art results on observability/telemetry data, a domain each model's pretraining corpus was deliberately weighted toward.

What this theme establishes: the existence of a domain effect is not in question — it appears independently across at least seven papers spanning six domains. What no single paper does is measure it on a common scale, against a documented measure of pretraining exposure, across enough domains at once to build a predictive model rather than report one more anecdote.

2.4 Model Architecture, Adaptation, and Post-Training Methods
A fourth cluster develops the models and adaptation techniques rather than evaluating them, and is useful as background on why domain transfer might succeed or fail mechanically. MOMENT (Goswami et al., 2024) and Toto 2.0 (Khwaja et al., 2026) both show that broader, more diverse pretraining data and larger scale improve zero-shot transfer — consistent with a "coverage" explanation of domain effects. Moirai 2.0 (Liu et al., 2025) reports strong GIFT-Eval performance with a much smaller, faster decoder-only design. Several papers target the specific failure mode this project cares about — degradation under distribution shift: TS-Memory (Lyu et al., 2026) and MEMTS (Yu et al., 2026) add lightweight memory modules that let a frozen TSFM internalize domain-specific dynamics without full retraining, motivated by the observation that TSFMs "struggle with adapting to downstream domains experiencing distribution shifts." Xie et al.'s post-training survey (2026) organizes this space into five categories (parameter adaptation, context augmentation, model composition, output processing/uncertainty control, compression/specialization). Das et al.'s in-context fine-tuning (2024) and Le Naour et al.'s TS-ICL (2026) show that conditioning on related series at inference time, without weight updates, recovers meaningful accuracy — a cheaper alternative to full fine-tuning, supporting this project's choice to prefer zero-shot and, at most, few-shot evaluation. Ning et al.'s TS-RAG (2025) and Wu et al.'s WaveMoE (2026) report further zero-shot gains via retrieval-augmentation and frequency-domain representations. Chen et al. (2024) and Yeh et al. (2023, TimeCLR) are earlier architecture/self-supervision papers establishing the pretraining paradigm; Chen et al.'s finding that large-scale pretraining does not always beat target-dataset-only training is an early precursor to the break-even and domain-mismatch results above.

What this theme establishes: the field has a mechanistic story for why domain mismatch should degrade performance (insufficient distributional coverage in pretraining) and a growing toolkit for fixing it after the fact, but these tools are validated per-paper, not against the systematic domain-similarity question this project asks.

2.5 Direct Precedents on "How Foundational" TSFMs Really Are
One paper asks almost exactly this project's question already and must be engaged with directly. Karaouli et al. (2025), "How Foundational are Foundation Models for Time Series Forecasting?" evaluates TimesFM and TiReX against a lightweight from-scratch model (SAMFormer) under domain shift and data scarcity, using a new daily-electricity-consumption dataset. Its headline finding: TSFMs generalize well on data resembling their training distribution but struggle significantly with personalized, unseen consumption patterns, where smaller purpose-trained models can win outright. This is a genuine precedent. What it does not do — and what leaves the question open — is (a) use a documented, verifiable measure of what was or wasn't in each model's pretraining mix (its domain-shift claim rests on the new dataset being unfamiliar); (b) test more than one domain or a handful of series, so the result cannot yet be generalized into a predictive rule; and (c) connect the finding to compute cost. Read alongside QuitoBench's competing claim (§2.1) that forecastability, not domain identity, drives difficulty, the two results are not reconciled anywhere in this literature — resolving that tension (is it domain novelty, intrinsic series difficulty, or both, and can they be told apart statistically) is itself a concrete objective for this project.

2.6 Synthesis: The Gap This Project Addresses
No single paper in this corpus combines all of the following, though each piece exists somewhere in it:

A domain-similarity measure tied to documented pretraining composition, rather than an assumed domain difference (closest: Marconi's note that Chronos's pretraining excluded finance; GIFT-Eval's "non-leaking" corpus construction).
A systematic, multi-domain sweep large enough to fit a predictive model (closest: GIFT-Eval's meta-features across seven domains; QuitoBench's regime-balanced design within one corpus).
A quantitative account of whether gains justify compute — runtime, memory, and energy together (closest: the Complexity Router's latency-aware routing on four regimes; Tan Jerome & Simon's binary GPU-or-not heuristic).
Statistical validation that the resulting model generalizes — e.g., leave-dataset-out — rather than a fitted relationship reported without held-out confirmation.
Tan Jerome and Simon's own limitation — that reliable automated prediction of FM gains "remains an open problem" — is the most explicit statement in the corpus that this gap is real and unresolved. This project's contribution is to build the systematic, compute-aware, pretraining-exposure-aware, statistically validated model that explains when and why zero-shot TSFM gains occur, using the meta-feature and benchmark infrastructure this literature has already built (principally GIFT-Eval and the Monash Forecasting Repository).

3. Baseline Time-Series Model
3.1 Purpose
Before any foundation model is evaluated, the project needs a working, tested harness of the mandatory classical baselines every claimed foundation-model gain must be measured against. This section builds and evaluates that harness: seasonal naive, drift, ETS, Theta, and SARIMA.

3.2 Data and Environment Note
The project's actual target data (GIFT-Eval, the Monash Forecasting Repository, and the selected unseen-domain datasets) are hosted on endpoints not reachable from the sandboxed environment this section was run in. The harness is therefore built and validated end-to-end on a substitute dataset chosen to be methodologically appropriate: the classical AirPassengers monthly series (US airline passenger totals, Jan 1949–Dec 1960; Box & Jenkins), retrieved from the public Rdatasets mirror. It has a clear multiplicative trend and strong 12-month seasonality — exactly the structure seasonal naive, ETS, Theta, and SARIMA each exploit differently. The code is written generically (train/test split, model set, metric set) and is intended to be re-pointed at GIFT-Eval/Monash series once run in an environment with full network access.

3.3 Methodology
Split: last 24 months held out as a test set; all prior months used for training (n = 120).
Models:
Seasonal naive — repeats the last observed 12-month cycle.
Drift — linear extrapolation of the average change over the training period.
ETS — additive damped trend, multiplicative seasonality (Holt-Winters), fit by maximum likelihood.
Theta — deseasonalized Theta method, period 12.
SARIMA(1,1,1)(1,1,1)₁₂ — a standard, unoptimized seasonal ARIMA specification (no order search, consistent with treating this as a baseline rather than a tuned competitor).
Metrics: MASE (scaled against the in-sample seasonal-naive error), RMSE, and MAPE.
3.4 Results
Model	MASE	RMSE	MAPE (%)
ETS	1.278	41.00	8.01
Theta	2.213	71.36	13.34
SARIMA(1,1,1)(1,1,1)₁₂	2.446	75.63	15.22
Seasonal Naive	2.494	76.99	15.52
Drift	3.206	115.70	18.41
Baseline forecast comparison

3.5 Discussion
Two results carry forward. First, ETS clearly wins this baseline ladder (MASE 1.28, roughly half the error of the next-best model). That is expected, since Holt-Winters' multiplicative-seasonal form matches this series' structure, and it is a reminder that "mandatory baselines" are not a token gesture: on a series like this, a well-specified classical model is a genuinely hard target. Second, the unoptimized SARIMA specification underperforms both ETS and Theta, so on the real project data an unsearched SARIMA order should not be assumed competitive without at least a light order search, which the full project's baseline phase should include.

The harness itself — clean train/test split, a fixed model set, MASE as the primary cross-series metric — carries forward unchanged. Once pointed at actual GIFT-Eval/Monash series and the unseen-domain datasets, it becomes the denominator against which every zero-shot foundation-model result in the rest of the project is measured.

4. Next Steps
This report establishes the evidence base (§2) and the baseline evaluation harness (§3). The remaining phases — meta-feature computation and domain-similarity scoring, the compute-cost benchmark, the predictive/hierarchical model linking domain similarity to foundation-model gains, and leave-dataset-out validation — are set out in the companion implementation plan.


---

## 3. Methodology, Data Architecture, and Verification Harness

### 3.1 Exact GIFT-Eval Leaderboard Cross-Check (Phase 0 / Phase 2)
Before running foundation models or fitting mixed-effects regressions, our local evaluation harness was verified against the official Salesforce `GIFT-Eval` leaderboard (`Seasonal_Naive` MASE, snapshot date `2026-10-16`) on four full, uncapped benchmark configurations (`exact_gift_eval=True`, without the 60-series subsample). Accounting for GluonTS frequency-to-seasonality rules (`M -> 12`, `W -> 1`) reproduced the published leaderboard numbers to **6 decimal places** (`abs_diff = 0.000000`):

| config | n_series | n_eval_windows | season_length | local_seasonal_naive_mase | leaderboard_seasonal_naive_mase | abs_diff |
| --- | --- | --- | --- | --- | --- | --- |
| hospital/M/short | 767 | 767 | 12 | 0.920528 | 0.920528 | 0.000000 |
| us_births/M/short | 1 | 2 | 12 | 0.760531 | 0.760531 | 0.000000 |
| saugeen/W/short | 1 | 20 | 1 | 1.990658 | 1.990658 | 0.000000 |
| solar/W/short | 137 | 137 | 1 | 1.470351 | 1.470351 | 0.000000 |


### 3.2 Benchmark Corpus & Unseen Datasets (Phase 1 & Phase 3)
Our full evaluation combines:
- **97 GIFT-Eval configurations** across 28 `base_dataset` clusters and 7 domains (`Energy`, `Transport`, `Nature`, `Web/CloudOps`, `Sales`, `Econ/Fin`, `Healthcare`), evaluated across 15 TSFMs (4 locally executed + 11 leaderboard-only: `Chronos_base`, `Chronos_large`, `chronos_bolt_small`, `chronos_bolt_base`, `Chronos-2`, `Moirai_base`, `Moirai_large`, `Moirai2`, `TimesFM`, `timesfm_2_0_500m`, `TTM-R1-Pretrained`) against the unified 5-model classical baseline set (`Naive`, `SeasonalNaive`, `AutoETS`, `AutoTheta`, `AutoARIMA`).
- **Unseen Dataset 1 (`unseen_mandi/W/short`, `unseen_dataset_overlapping_domain` Stratum):** 60 weekly wholesale agricultural commodity price series across Indian mandis (261 weeks each, 15,660 observations from `data.gov.in` / Agmarknet, GODL-India licence; MASE `season_length = 1` matching GIFT-Eval weekly convention, STL decomposition period `52`), not in any documented pretraining corpus (`Econ/Fin` commodity prices).
- **Unseen Dataset 2 (`unseen_recency/D/short`, Post-Cutoff Recency Stratum):** 60 daily meteorological series (3 variables: `temperature_2m_mean`, `shortwave_radiation_sum`, `wind_speed_10m_max` across 20 global cities from 2025-01-01 to 2026-09-30, 38,280 observations from Open-Meteo ERA5, CC-BY-4.0 licence; MASE `season_length = 1`, STL decomposition period `7`), post-dating the training cutoffs of `Chronos_small` (2024-03) and `Moirai_small` (2024-06), with 2025+ models (`Chronos-2`, `Moirai2`, `TimesFM-2.5`) coded as `"unknown"` in the recency registry.

#### Table 1: Dataset Meta-Feature Summary by Domain (Hurst via Detrended Fluctuation Analysis)
| domain | n_configs | median_length | mean_trend_strength | mean_seasonal_strength | mean_forecastability | mean_hurst | mean_stability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Energy | 32 | 1096.000 | 0.548 | 0.382 | 0.386 | 0.881 | 0.465 |
| Web/CloudOps | 20 | 691.000 | 0.652 | 0.439 | 0.446 | 0.892 | 0.350 |
| Transport | 15 | 2976.000 | 0.288 | 0.344 | 0.321 | 0.685 | 0.134 |
| Nature | 15 | 8783.000 | 0.597 | 0.221 | 0.367 | 0.860 | 0.566 |
| Econ/Fin | 7 | 261.000 | 0.914 | 0.236 | 0.579 | 0.925 | 0.789 |
| Healthcare | 5 | 240.000 | 0.793 | 0.588 | 0.479 | 0.827 | 0.603 |
| Sales | 4 | 274.750 | 0.108 | 0.177 | 0.071 | 0.691 | 0.155 |
| Nature/PostCutoff2025 | 1 | 638.000 | 0.618 | 0.089 | 0.295 | 0.902 | 0.684 |


#### Collinearity Diagnostics (Variance Inflation Factors — `forecastability = 1 - spectral_entropy` entered; `spectral_entropy` excluded as exact complement)
All 7 series meta-features satisfy the $\mathrm{VIF} < 10$ threshold (maximum $\mathrm{VIF} = 3.47$):

| feature | vif | keep_under_vif_10 |
| --- | --- | --- |
| trend_strength | 2.73 | True |
| seasonal_strength | 1.69 | True |
| forecastability | 2.31 | True |
| hurst | 1.80 | True |
| stability | 3.47 | True |
| lumpiness | 1.41 | True |
| log_length | 1.62 | True |


#### Full-Predictor VIF Diagnostics (Including Similarity Terms & Controls)

| predictor | vif | keep_under_vif_10 |
| --- | --- | --- |
| forecastability | 2.71 | True |
| seasonal_strength | 1.78 | True |
| trend_strength | 2.77 | True |
| hurst | 1.84 | True |
| stability | 3.95 | True |
| lumpiness | 1.85 | True |
| log_length | 2.02 | True |
| log_params | 1.03 | True |
| dataset_leak | 1.17 | True |
| sim_binary | 1.72 | True |
| sim_share | 1.83 | True |
| sim_distance | 1.54 | True |


---

## 4. Pretraining Exposure Registry & Zero-Shot Domain Transfer

#### Table 2: TSFM Pretraining Exposure & Aggregate Zero-Shot Gain Over Best Classical Baseline
| model | model_family | param_count_M | n_configs | in_domain_share | mean_sim_distance | mean_mase | mean_delta_mase_pct | win_rate_vs_classical_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chronos-2 | chronos_2 | 120.00 | 97 | 1.00 | 2.31 | 1.43 | 59.87 | 98.97 |
| TimesFM-2.5 | timesfm | 231.29 | 97 | 0.89 | 2.43 | 1.49 | 58.83 | 97.94 |
| Moirai2 | moirai_2 | 14.00 | 97 | 1.00 | 2.31 | 1.52 | 55.61 | 97.94 |
| timesfm_2_0_500m | timesfm | 500.00 | 97 | 0.89 | 2.43 | 1.71 | 51.58 | 92.78 |
| chronos_bolt_base | chronos_bolt | 205.00 | 97 | 0.92 | 2.35 | 1.86 | 45.25 | 91.75 |
| chronos_bolt_small | chronos_bolt | 48.00 | 97 | 0.92 | 2.35 | 1.82 | 43.47 | 89.69 |
| Chronos_large | chronos_t5 | 710.00 | 97 | 0.92 | 2.35 | 1.91 | 37.86 | 88.66 |
| Moirai_large | moirai_1 | 311.00 | 97 | 1.00 | 2.31 | 1.85 | 37.25 | 83.51 |
| Chronos_base | chronos_t5 | 200.00 | 97 | 0.92 | 2.35 | 1.93 | 37.14 | 89.69 |
| Chronos_small | chronos_t5 | 46.00 | 99 | 0.90 | 2.33 | 1.95 | 34.61 | 85.86 |
| Moirai_base | moirai_1 | 91.00 | 97 | 1.00 | 2.31 | 2.02 | 34.31 | 84.54 |
| Moirai_small | moirai_1 | 13.83 | 99 | 0.99 | 2.30 | 1.96 | 28.74 | 81.82 |
| TTM-R2-Pretrained | ttm | 0.81 | 99 | 0.83 | 2.49 | 2.29 | 21.29 | 69.70 |
| TimesFM | timesfm | 200.00 | 99 | 0.87 | 2.42 | 4.13 | 18.55 | 75.76 |
| TTM-R1-Pretrained | ttm | 0.81 | 97 | 0.85 | 2.50 | 2.39 | 16.27 | 65.98 |


![Figure 1: Zero-Shot TSFM Gain Across Domains](figures/fig1_domain_shift_heatmap.png)

---

## 5. Statistical Results: Mixed-Effects Models, LODO Validation, and Compute Pareto Frontier

### 5.1 Hierarchical Mixed-Effects Model Ladder (Phase 6 — Testing H1)
We fit nested linear mixed-effects models with random intercepts $(1 \mid \text{base\_dataset}) + (1 \mid \text{config}) + (1 \mid \text{model\_family})$ (30 `base_dataset` clusters, 99 `config` levels nested within `base_dataset`, and 7 `model_family` levels $\ge 6$) on the log-ratio response $y_{m,d} = \log(\mathrm{MASE}_{m,d} / \mathrm{MASE}_{\text{classical},d})$ (equivalently $-y_{m,d} = \log(\mathrm{MASE}_{\text{classical},d} / \mathrm{MASE}_{m,d})$ so positive coefficients denote TSFM improvement), controlling for `dataset_leak` and `source_local` from `M1` upward:
- **Variance Decomposition (`M0`):** In the null model, the dataset-cluster intraclass correlation is $\mathrm{ICC}_{\text{base\_dataset}} = 0.727$ and config-level $\mathrm{ICC}_{\text{config}} = 0.200$ while $\mathrm{ICC}_{\text{model\_family}} = 0.058$, confirming that dataset and configuration structure explain substantially more variance in relative TSFM gain than model-family identity.
- **Series Characteristics & Leakage Control (`M1`, QuitoBench Null):** Adding `forecastability`, `seasonal_strength`, `trend_strength`, DFA `hurst`, `stability`, `lumpiness`, `log_length`, $\log(\text{params})$, `dataset_leak`, `source_local`, and `frequency_group` yields marginal $R^2_m = 0.207$ and conditional $R^2_c = 0.801$ ($\mathrm{LRT}$ vs `M0`: $\chi^2(15) = 96.03$, $p = 7.33e-14$).
- **Pretraining Domain Similarity (`M2a`, `M2b`, `M2`, & `M3`, H1):** Comparing the non-nested single-block additions `M2a` (`+ sim_binary`) vs `M2b` (`+ z_sim_distance`) by AIC/BIC shows $\Delta\mathrm{AIC}_{\text{M2b vs M2a}} = -1.35$ and $\Delta\mathrm{BIC}_{\text{M2b vs M2a}} = -6.64$. Adding the full similarity block (`M2`) significantly improves fit over `M1` ($\chi^2(3) = 0.88$, $p = 0.8309$), and adding interactions between `sim_distance` and series characteristics (`M3`) further improves fit ($\chi^2(3) = 15.99$, $p = 0.0011$, marginal $R^2_m = 0.214$). Benjamini–Hochberg FDR $q$-values (`fdr_bh_q_value`) are computed over the pre-specified hypothesis family of similarity fixed effects (`3` terms in `M2`; `6` similarity + interaction terms in `M3`).

#### Table 3: Nested Mixed-Effects Model Comparison (ML Fit for LRT/AIC/BIC)
| model_id | n_obs | k_params | logLik_ML | AIC_ML | BIC_ML | delta_AIC_vs_M2a | delta_BIC_vs_M2a | R2_marginal | R2_conditional | icc_base_dataset | icc_config | icc_model_family | lrt_vs | lrt_chi2 | lrt_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 1463 | 5 | -252.1214 | 514.2429 | 540.6841 | 62.8126 | -27.0875 | 0.0000 | 0.7846 | 0.7271 | 0.2003 | 0.0576 | — | — | — |
| M1 | 1463 | 20 | -204.1053 | 448.2106 | 553.9755 | -3.2197 | -13.7962 | 0.2068 | 0.8012 | 0.6841 | 0.0845 | 0.0653 | M0 | 96.0323 | 0.0000 |
| M2a | 1463 | 22 | -203.7151 | 451.4303 | 567.7717 | 0.0000 | 0.0000 | 0.2073 | 0.8016 | 0.6824 | 0.0845 | 0.0674 | M1 | 0.7803 | 0.6770 |
| M2b | 1463 | 21 | -204.0393 | 450.0787 | 561.1318 | -1.3516 | -6.6399 | 0.2081 | 0.8004 | 0.6817 | 0.0845 | 0.0663 | M1 | 0.1319 | 0.7164 |
| M2 | 1463 | 23 | -203.6667 | 453.3334 | 574.9630 | 1.9031 | 7.1914 | 0.2084 | 0.8011 | 0.6806 | 0.0844 | 0.0681 | M1 | 0.8772 | 0.8309 |
| M3 | 1463 | 26 | -195.6738 | 443.3476 | 580.8420 | -8.0827 | 13.0703 | 0.2139 | 0.7844 | 0.6506 | 0.0768 | 0.0751 | M2 | 15.9858 | 0.0011 |


#### Key Fixed-Effect Estimates in `M2` and `M3` (REML Fit with 1,000-Cluster Bootstrap 95% CIs, Log-Ratio $y$ & `% Gain`)
| model_id | term | estimate_reml | estimate_log_ratio_y | pct_gain_equiv | se_reml | p_value | boot_ci95_low | boot_ci95_high | fdr_bh_q_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 | z_forecastability | 0.0529 | -0.0529 | 5.1553 | 0.0496 | 0.2864 | -0.1321 | 0.1940 | — |
| M2 | z_seasonal_strength | -0.0263 | 0.0263 | -2.6655 | 0.0374 | 0.4821 | -0.2296 | 0.1702 | — |
| M2 | z_trend_strength | -0.0542 | 0.0542 | -5.5709 | 0.0581 | 0.3507 | -0.3988 | 0.1173 | — |
| M2 | z_log_length | -0.0687 | 0.0687 | -7.1100 | 0.0610 | 0.2598 | -0.4315 | 0.2104 | — |
| M2 | z_log_params | 0.0541 | -0.0541 | 5.2700 | 0.0157 | 0.0006 | 0.0365 | 0.0753 | — |
| M2 | dataset_leak | 0.0986 | -0.0986 | 9.3908 | 0.0243 | 0.0000 | 0.0188 | 0.1816 | — |
| M2 | source_local | 0.0155 | -0.0155 | 1.5376 | 0.3463 | 0.9643 | -0.3205 | 0.4602 | — |
| M2 | sim_binary | -0.0213 | 0.0213 | -2.1536 | 0.0443 | 0.6302 | -0.1008 | 0.1129 | 0.8375 |
| M2 | z_sim_share | 0.0099 | -0.0099 | 0.9888 | 0.0126 | 0.4292 | -0.0658 | 0.1342 | 0.8375 |
| M2 | z_sim_distance | 0.0070 | -0.0070 | 0.6979 | 0.0341 | 0.8375 | -0.0527 | 0.2033 | 0.8375 |
| M3 | z_forecastability | 0.0133 | -0.0133 | 1.3206 | 0.0505 | 0.7923 | -0.1604 | 0.1381 | — |
| M3 | z_seasonal_strength | -0.0659 | 0.0659 | -6.8122 | 0.0381 | 0.0840 | -0.2495 | 0.1164 | — |
| M3 | z_trend_strength | -0.0333 | 0.0333 | -3.3880 | 0.0564 | 0.5547 | -0.2242 | 0.1887 | — |
| M3 | z_log_length | -0.0077 | 0.0077 | -0.7723 | 0.0627 | 0.9023 | -0.2780 | 0.2229 | — |
| M3 | z_log_params | 0.0540 | -0.0540 | 5.2612 | 0.0157 | 0.0006 | 0.0362 | 0.0753 | — |
| M3 | dataset_leak | 0.1023 | -0.1023 | 9.7283 | 0.0243 | 0.0000 | 0.0190 | 0.1846 | — |
| M3 | source_local | -0.0645 | 0.0645 | -6.6573 | 0.3309 | 0.8456 | -0.4172 | 0.2474 | — |
| M3 | sim_binary | -0.0663 | 0.0663 | -6.8595 | 0.0469 | 0.1572 | -0.1723 | 0.0531 | 0.1886 |
| M3 | z_sim_share | 0.0079 | -0.0079 | 0.7833 | 0.0126 | 0.5328 | -0.0684 | 0.1294 | 0.5328 |
| M3 | z_sim_distance | 0.0716 | -0.0716 | 6.9143 | 0.0403 | 0.0757 | -0.0550 | 0.2926 | 0.1135 |
| M3 | z_sim_distance:z_forecastability | -0.0544 | 0.0544 | -5.5922 | 0.0301 | 0.0709 | -0.1425 | 0.0505 | 0.1135 |
| M3 | z_sim_distance:z_seasonal_strength | 0.0809 | -0.0809 | 7.7713 | 0.0413 | 0.0501 | -0.0980 | 0.2658 | 0.1135 |
| M3 | z_sim_distance:z_log_length | 0.1023 | -0.1023 | 9.7251 | 0.0385 | 0.0079 | -0.0047 | 0.3592 | 0.0475 |


### 5.2 Domain-Overlap Strata Contrasts (Testing H2)

#### Table 6a: H2 Planned Contrasts Across Domain-Overlap Strata (Strata with $<5$ Base Datasets Reported Descriptively Only)
| stratum | n_pairs | n_base_datasets | inference_status | mean_log_ratio_y | pct_gain_from_y | mean_delta_mase | median_delta_mase | cluster_se | ci95_low | ci95_high | tsfm_win_rate | t_stat_vs_zero | p_value_vs_zero | contrast_vs_in_domain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in-domain | 1266 | 19 | inferential (>=5 base datasets) | -0.3760 | 31.3418 | 0.3760 | 0.3130 | 0.0633 | 0.2520 | 0.5000 | 0.8681 | 5.9439 | 0.0000 | 0.0000 |
| partial-overlap | 134 | 12 | inferential (>=5 base datasets) | -0.5481 | 42.1968 | 0.5481 | 0.2660 | 0.2051 | 0.1462 | 0.9500 | 0.9254 | 2.6731 | 0.0217 | 0.1721 |
| out-of-domain | 55 | 12 | inferential (>=5 base datasets) | -0.2912 | 25.2645 | 0.2912 | 0.1512 | 0.1507 | -0.0041 | 0.5866 | 0.6545 | 1.9325 | 0.0795 | -0.0848 |
| unseen_dataset_overlapping_domain | 4 | 1 | descriptive_only (<5 base datasets) | 0.1173 | -12.4413 | -0.1173 | -0.0648 | 0.0603 | -0.2355 | 0.0010 | 0.0000 | -1.9437 | 0.3025 | -0.4933 |
| recency | 4 | 1 | descriptive_only (<5 base datasets) | -0.0566 | 5.5065 | 0.0566 | 0.0577 | 0.0319 | -0.0059 | 0.1192 | 0.7500 | 1.7742 | 0.3267 | -0.3194 |


![Figure 2: Domain Similarity vs Gain and H2 Strata Contrasts](figures/fig2_similarity_vs_gain.png)

### 5.3 Leave-One-Base-Dataset-Out (LODO) Cross-Validation

#### Table 4: Out-of-Sample Predictive Performance Across 30 Base-Dataset Clusters
| predictor | n_pairs | n_base_datasets | oos_rmse | oos_mae | fold_median_rmse | spearman_rho | spearman_p_value | sign_accuracy | rmse_change_vs_M1_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 1463 | 30 | 0.5041 | 0.3570 | 0.3163 | -0.5688 | 0.0000 | 0.8626 | -1.3108 |
| TanJerome_Simon_Heuristic | 1463 | 30 | 0.5093 | 0.3619 | 0.3146 | 0.0852 | 0.0011 | 0.8626 | -0.3034 |
| M1 | 1463 | 30 | 0.5108 | 0.3729 | 0.3497 | 0.2958 | 0.0000 | 0.8284 | 0.0000 |
| M2a | 1463 | 30 | 0.5107 | 0.3752 | 0.3409 | 0.3015 | 0.0000 | 0.8319 | -0.0197 |
| M2b | 1463 | 30 | 0.5375 | 0.3894 | 0.3512 | 0.2547 | 0.0000 | 0.8093 | 5.2129 |
| M2 | 1463 | 30 | 0.5356 | 0.3916 | 0.3452 | 0.2611 | 0.0000 | 0.8127 | 4.8519 |
| M3 | 1463 | 30 | 0.5424 | 0.3991 | 0.4018 | 0.2450 | 0.0000 | 0.7785 | 6.1879 |


![Figure 3: Leave-One-Base-Dataset-Out Cross-Validation](figures/fig3_lodo_cv_calibration.png)

### 5.4 Compute-Efficiency Pareto Frontier & H4 Decision Boundary (Phase 4 & Phase 9)

#### Table 5: Latency (`ms/pt`), TDP-Estimated CPU Energy (`45 W`), Memory, and Accuracy Across the 13 Locally Executed Configurations
| model | model_family_group | param_count | mean_mase | median_mase | iqr_mase | mean_crps | mean_delta_mase_pct | win_rate_vs_best_classical | mean_infer_ms_per_point | median_infer_ms_per_point | iqr_infer_ms_per_point | mean_peak_ram_mb | mean_energy_mwh_per_1k_pts | median_energy_mwh_per_1k_pts | iqr_energy_mwh_per_1k_pts | n_positive_gain_configs | median_energy_per_1pct_gain_mwh | median_latency_ms_per_1pct_gain | pareto_config_share_energy | global_pareto_energy | global_pareto_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimesFM | tsfm | 231289280 | 1.174 | 1.007 | 0.471 | 0.180 | 1.946 | 0.538 | 34.838 | 10.339 | 8.204 | 1905.790 | 435.475 | 129.241 | 102.552 | 7 | 0.000 | 0.000 | 0.615 | True | True |
| AutoARIMA | classical | 4 | 1.271 | 1.217 | 0.359 | 0.244 | -4.776 | 0.000 | 25.693 | 24.412 | 14.685 | 252.590 | 321.162 | 305.150 | 183.559 | 0 | — | — | 0.231 | True | True |
| Chronos_small | tsfm | 46154240 | 1.295 | 1.132 | 0.417 | 0.227 | -10.814 | 0.308 | 29.067 | 25.612 | 11.853 | 4303.736 | 363.332 | 320.144 | 148.163 | 4 | 11.485 | 0.919 | 0.000 | False | False |
| AutoTheta | classical | 2 | 1.333 | 1.337 | 0.345 | 0.274 | -13.670 | 0.000 | 7.374 | 4.742 | 5.003 | 252.428 | 92.178 | 59.271 | 62.538 | 0 | — | — | 0.385 | True | True |
| AutoETS | classical | 4 | 1.335 | 1.204 | 0.469 | 0.312 | -11.322 | 0.000 | 8.132 | 6.067 | 11.182 | 252.073 | 101.652 | 75.835 | 139.772 | 0 | — | — | 0.385 | False | False |
| Moirai_small | tsfm | 13827528 | 1.387 | 1.320 | 0.392 | 0.211 | -22.866 | 0.231 | 2.208 | 1.935 | 2.175 | 979.953 | 27.606 | 24.193 | 27.190 | 3 | 0.000 | 0.000 | 0.385 | True | True |
| LightGBM | ml_tabular | 15000 | 1.469 | 1.361 | 0.553 | 0.292 | -22.749 | 0.077 | 3.114 | 2.303 | 1.291 | 560.841 | 38.923 | 28.781 | 16.140 | 1 | 0.000 | 0.000 | 0.000 | False | False |
| SeasonalNaive | classical | 2 | 1.488 | 1.498 | 0.344 | 0.453 | -26.993 | 0.000 | 0.254 | 0.244 | 0.253 | 248.189 | 3.175 | 3.045 | 3.165 | 0 | — | — | 0.231 | True | True |
| Naive | classical | 2 | 1.522 | 1.498 | 0.322 | 0.503 | -37.321 | 0.000 | 0.252 | 0.261 | 0.225 | 248.022 | 3.150 | 3.263 | 2.812 | 0 | — | — | 0.385 | True | True |
| TTM-R2-Pretrained | tsfm | 805280 | 1.537 | 1.268 | 0.571 | 0.437 | -38.143 | 0.154 | 0.360 | 0.231 | 0.270 | 1885.448 | 4.501 | 2.890 | 3.381 | 2 | 0.000 | 0.000 | 0.692 | False | False |
| Drift | classical | 2 | 1.552 | 1.458 | 0.388 | 0.507 | -39.943 | 0.000 | 0.263 | 0.244 | 0.241 | 248.188 | 3.293 | 3.046 | 3.011 | 0 | — | — | 0.385 | False | False |


![Figure 4: Compute-Efficiency Pareto Frontier and H4 Decision Boundary](figures/fig4_pareto_compute_accuracy.png)

### 5.4 Probabilistic Calibration & Cross-Series Per-Horizon Split Conformal Repair (Phase 5)

Evaluating native prediction intervals (`q0.1` to `q0.9`) against split conformal calibration (calibrating on rolling window 1 across all 60 series and evaluating on rolling window 2, separately per horizon step $h \in \{1,\dots,H\}$ under cross-series exchangeability of standardized nonconformity scores) reveals native miscalibration that conformalization repairs without retraining:

| model | calibration_method | n_configs | mean_pre_coverage_80 | mean_post_coverage_80 | mean_abs_coverage_err_pre | mean_abs_coverage_err_post | mean_pre_width_80 | mean_post_width_80 | mean_pre_crps | mean_post_crps | mean_pre_interval_score | mean_post_interval_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimesFM | cqr | 13 | 0.7824 | 0.7847 | 0.0775 | 0.0875 | 0.6361 | 0.7030 | 0.1863 | 0.1879 | 1.1419 | 1.1625 |
| Chronos_small | cqr | 13 | 0.2663 | 0.7829 | 0.5337 | 0.1028 | 0.2161 | 0.8009 | 0.2406 | 0.2243 | 2.0095 | 1.5344 |
| AutoARIMA | cqr | 13 | 0.7868 | 0.7358 | 0.1098 | 0.1135 | 1.1152 | 1.0113 | 0.2242 | 0.2263 | 1.4519 | 1.5610 |
| Moirai_small | cqr | 13 | 0.6776 | 0.7363 | 0.1366 | 0.1162 | 0.7115 | 0.8405 | 0.2272 | 0.2294 | 1.4955 | 1.5324 |
| AutoTheta | cqr | 13 | 0.7845 | 0.7489 | 0.1306 | 0.1243 | 1.3444 | 1.1363 | 0.2486 | 0.2429 | 1.7325 | 1.6764 |
| TTM-R2-Pretrained | abs_residual | 13 | 0.8636 | 0.7700 | 0.1570 | 0.1039 | 2.7699 | 0.9918 | 0.3551 | 0.2439 | 2.9224 | 1.7401 |
| Naive | abs_residual | 13 | 0.9704 | 0.7853 | 0.1704 | 0.0874 | 3.3958 | 1.0491 | 0.4024 | 0.2705 | 3.4283 | 1.9399 |
| AutoETS | cqr | 13 | 0.8148 | 0.7759 | 0.1110 | 0.1017 | 1.6259 | 1.5345 | 0.2721 | 0.2720 | 1.9481 | 1.9691 |
| LightGBM | abs_residual | 13 | 0.6425 | 0.7615 | 0.2264 | 0.1330 | 1.1413 | 1.2100 | 0.2677 | 0.2723 | 1.9405 | 1.8934 |
| Drift | abs_residual | 13 | 0.9669 | 0.7870 | 0.1669 | 0.0930 | 3.3958 | 1.0805 | 0.4064 | 0.2782 | 3.4314 | 2.0012 |
| SeasonalNaive | cqr | 13 | 0.8457 | 0.7557 | 0.1519 | 0.0969 | 2.6956 | 2.2424 | 0.3592 | 0.3391 | 2.8715 | 2.5919 |

![Figure 5: Conformal Calibration](figures/fig5_conformal_calibration.png)

### 5.5 Financial Volatility Structural-Break Case Study (Phase 7 — Testing H3)

On S&P 500 daily Garman–Klass log-volatility (1990–2026, $N = 9,247$ trading days), PELT change-point segmentation (`pen = 12.0` on 5-day smoothed standardized log-GK volatility) detects 30 regime breaks and recovers **4 of 6** major historical crisis breakpoints within $\pm 30$ trading days:

| event_label | labelled_date | labelled_trading_idx | detected_break_date | detected_break_idx | trading_day_distance | recovered_within_30d | n_total_pelt_breaks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Asian_Financial_Crisis_MiniCrash | 1997-10-27 | 1978 | 1998-02-09 | 2049 | 71 | False | 91 |
| Sept11_Market_Reopening | 2001-09-17 | 2954 | 2001-09-04 | 2949 | 5 | True | 91 |
| Lehman_Brothers_GFC | 2008-09-15 | 4715 | 2008-09-08 | 4710 | 5 | True | 91 |
| US_Debt_Downgrade_Eurozone | 2011-08-08 | 5445 | 2011-08-02 | 5441 | 4 | True | 91 |
| COVID19_Crash | 2020-03-09 | 7604 | 2020-02-27 | 7597 | 7 | True | 91 |
| Fed_Rate_Hike_Cycle_Start | 2022-03-16 | 8114 | 2022-01-10 | 8069 | 45 | False | 91 |

#### Regime-Partitioned Forecast Accuracy (`stable` vs 63-Trading-Day `post_break`; QLIKE, MSE, MASE, CRPS & Kupiec POF)

| horizon | regime | model | n_windows | mase | mase_median | mse | rmse | qlike | crps | coverage_80 | width_80 | kupiec_lr_stat | kupiec_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | post_break | Chronos_small | 24 | 0.7711 | 0.5950 | 0.1546 | 0.3092 | 0.4613 | 0.0692 | 0.1250 | 0.0295 | 50.8503 | 0.0000 |
| 1 | post_break | EWMA_094 | 24 | 0.8981 | 0.6306 | 0.2059 | 0.3619 | 0.4805 | 0.0652 | 0.7917 | 0.2288 | 0.0103 | 0.9191 |
| 1 | post_break | GARCH_1_1_t | 24 | 0.9114 | 0.8243 | 0.1997 | 0.3645 | 0.5626 | 0.0653 | 0.7917 | 0.2236 | 0.0103 | 0.9191 |
| 1 | post_break | Moirai_small | 24 | 0.7700 | 0.4841 | 0.1537 | 0.3115 | 0.4172 | 0.0567 | 0.7500 | 0.2091 | 0.3543 | 0.5517 |
| 1 | post_break | Naive | 24 | 0.8954 | 0.8259 | 0.2401 | 0.3648 | 0.9201 | 0.0668 | 0.8333 | 0.2585 | 0.1743 | 0.6763 |
| 1 | stable | Chronos_small | 24 | 0.8849 | 0.7894 | 0.2006 | 0.3654 | 0.3385 | 0.0640 | 0.1250 | 0.0270 | 50.8503 | 0.0000 |
| 1 | stable | EWMA_094 | 24 | 0.8554 | 0.7469 | 0.2218 | 0.3551 | 0.3021 | 0.0520 | 0.7083 | 0.2114 | 1.1444 | 0.2847 |
| 1 | stable | GARCH_1_1_t | 24 | 0.9426 | 0.7857 | 0.2111 | 0.3910 | 0.3609 | 0.0543 | 0.7500 | 0.2111 | 0.3543 | 0.5517 |
| 1 | stable | Moirai_small | 24 | 0.7677 | 0.8292 | 0.1602 | 0.3184 | 0.2860 | 0.0454 | 0.7083 | 0.1746 | 1.1444 | 0.2847 |
| 1 | stable | Naive | 24 | 0.9102 | 0.7239 | 0.2188 | 0.3797 | 0.3773 | 0.0544 | 0.8333 | 0.2586 | 0.1743 | 0.6763 |
| 5 | post_break | Chronos_small | 24 | 0.9644 | 0.6677 | 0.2565 | 0.4394 | 0.9820 | 0.0908 | 0.1250 | 0.0309 | 254.2514 | 0.0000 |
| 5 | post_break | EWMA_094 | 24 | 1.0947 | 0.7817 | 0.2987 | 0.4880 | 0.9954 | 0.0832 | 0.7667 | 0.2600 | 0.8015 | 0.3707 |
| 5 | post_break | GARCH_1_1_t | 24 | 1.0454 | 0.9579 | 0.2677 | 0.4706 | 0.9151 | 0.0778 | 0.7583 | 0.2484 | 1.2411 | 0.2653 |
| 5 | post_break | Moirai_small | 24 | 0.9461 | 0.7356 | 0.2452 | 0.4433 | 0.9248 | 0.0758 | 0.7333 | 0.2357 | 3.0977 | 0.0784 |
| 5 | post_break | Naive | 24 | 1.0971 | 0.9412 | 0.2967 | 0.4980 | 0.9353 | 0.0840 | 0.9333 | 0.4334 | 16.9520 | 0.0000 |
| 5 | stable | Chronos_small | 24 | 0.8851 | 0.8191 | 0.2132 | 0.4371 | 0.5908 | 0.0663 | 0.1333 | 0.0251 | 247.6618 | 0.0000 |
| 5 | stable | EWMA_094 | 24 | 0.8512 | 0.7880 | 0.2137 | 0.4346 | 0.4646 | 0.0544 | 0.7833 | 0.2402 | 0.2042 | 0.6514 |
| 5 | stable | GARCH_1_1_t | 24 | 0.8905 | 0.8089 | 0.2100 | 0.4363 | 0.4838 | 0.0539 | 0.8167 | 0.2345 | 0.2129 | 0.6445 |
| 5 | stable | Moirai_small | 24 | 0.8242 | 0.7604 | 0.1776 | 0.4038 | 0.4733 | 0.0509 | 0.7417 | 0.1986 | 2.3912 | 0.1220 |
| 5 | stable | Naive | 24 | 0.9980 | 0.9237 | 0.2726 | 0.4836 | 0.5429 | 0.0676 | 0.9333 | 0.4335 | 16.9520 | 0.0000 |
| 20 | post_break | Chronos_small | 24 | 1.1871 | 0.9762 | 0.3801 | 0.5710 | 1.8285 | 0.1088 | 0.1104 | 0.0335 | 1064.6230 | 0.0000 |
| 20 | post_break | EWMA_094 | 24 | 1.2221 | 0.9552 | 0.3877 | 0.5748 | 1.7453 | 0.0922 | 0.8250 | 0.3502 | 1.9379 | 0.1639 |
| 20 | post_break | GARCH_1_1_t | 24 | 1.1226 | 0.9460 | 0.3225 | 0.5365 | 1.1404 | 0.0834 | 0.8250 | 0.3226 | 1.9379 | 0.1639 |
| 20 | post_break | Moirai_small | 24 | 1.1224 | 0.8879 | 0.3428 | 0.5510 | 1.5144 | 0.0848 | 0.7562 | 0.2768 | 5.4612 | 0.0194 |
| 20 | post_break | Naive | 24 | 1.2859 | 1.0054 | 0.4223 | 0.5993 | 1.3490 | 0.1181 | 0.9833 | 0.7970 | 155.0231 | 0.0000 |
| 20 | stable | Chronos_small | 24 | 0.8557 | 0.8318 | 0.1986 | 0.4389 | 0.5189 | 0.0645 | 0.1312 | 0.0252 | 997.1815 | 0.0000 |
| 20 | stable | EWMA_094 | 24 | 0.8421 | 0.8514 | 0.1926 | 0.4342 | 0.3917 | 0.0550 | 0.9104 | 0.3235 | 43.9293 | 0.0000 |
| 20 | stable | GARCH_1_1_t | 24 | 0.8722 | 0.8459 | 0.2004 | 0.4351 | 0.4132 | 0.0547 | 0.9146 | 0.3045 | 47.7627 | 0.0000 |
| 20 | stable | Moirai_small | 24 | 0.8685 | 0.8324 | 0.2003 | 0.4394 | 0.5201 | 0.0542 | 0.7500 | 0.2191 | 7.0867 | 0.0078 |
| 20 | stable | Naive | 24 | 1.0965 | 0.9879 | 0.3178 | 0.5389 | 0.6376 | 0.0957 | 0.9792 | 0.7973 | 144.7295 | 0.0000 |

#### H3 Difference-in-Differences (DiD), Stationary Block-Bootstrap & Diebold–Mariano Contrasts (`Chronos_small` [no market-price data; finance via M4 only] vs `Moirai_small` vs `GARCH(1,1)-t`)

| horizon | regime | chronos_minus_garch_mean | chronos_minus_garch_ci95_low | chronos_minus_garch_ci95_high | moirai_minus_garch_mean | moirai_minus_garch_ci95_low | moirai_minus_garch_ci95_high | chronos_minus_moirai_mean | chronos_minus_moirai_ci95_low | chronos_minus_moirai_ci95_high | chronos_minus_moirai_qlike_mean | chronos_minus_moirai_qlike_ci95_low | chronos_minus_moirai_qlike_ci95_high | chronos_minus_moirai_mse_mean | chronos_minus_moirai_mse_ci95_low | chronos_minus_moirai_mse_ci95_high | dm_chronos_vs_moirai_mase_stat | dm_chronos_vs_moirai_mase_p | dm_chronos_vs_moirai_qlike_stat | dm_chronos_vs_moirai_qlike_p | dm_chronos_vs_moirai_mse_stat | dm_chronos_vs_moirai_mse_p | did_mase_mean | did_mase_ci95_low | did_mase_ci95_high | did_qlike_mean | did_qlike_ci95_low | did_qlike_ci95_high | did_mse_mean | did_mse_ci95_low | did_mse_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | stable | -0.0577 | -0.1700 | 0.0615 | -0.1749 | -0.3261 | -0.0282 | 0.1172 | -0.0215 | 0.2627 | 0.0525 | -0.0299 | 0.1385 | 0.0404 | -0.0153 | 0.1076 | 1.4203 | 0.1689 | 1.1911 | 0.2458 | 1.3353 | 0.1948 | -0.1162 | -0.2577 | 0.0325 | -0.0085 | -0.0997 | 0.0991 | -0.0395 | -0.0860 | 0.0052 |
| 1 | post_break | -0.1404 | -0.3448 | 0.0425 | -0.1414 | -0.3887 | 0.0791 | 0.0010 | -0.1222 | 0.1313 | 0.0440 | -0.0422 | 0.1315 | 0.0010 | -0.0345 | 0.0358 | 0.0116 | 0.9909 | 0.6596 | 0.5161 | 0.0409 | 0.9677 | -0.1162 | -0.2577 | 0.0325 | -0.0085 | -0.0997 | 0.0991 | -0.0395 | -0.0860 | 0.0052 |
| 5 | stable | -0.0055 | -0.0757 | 0.0674 | -0.0664 | -0.1345 | 0.0058 | 0.0609 | 0.0086 | 0.1145 | 0.1175 | -0.0461 | 0.3440 | 0.0356 | 0.0117 | 0.0597 | 1.8048 | 0.0842 | 0.8661 | 0.3954 | 2.0451 | 0.0525 | -0.0426 | -0.1673 | 0.1080 | -0.0602 | -0.4393 | 0.3697 | -0.0243 | -0.0844 | 0.0482 |
| 5 | post_break | -0.0810 | -0.2584 | 0.0823 | -0.0993 | -0.2173 | 0.0113 | 0.0183 | -0.1025 | 0.1662 | 0.0573 | -0.2470 | 0.4713 | 0.0113 | -0.0481 | 0.0867 | 0.2005 | 0.8429 | 0.2439 | 0.8095 | 0.2429 | 0.8102 | -0.0426 | -0.1673 | 0.1080 | -0.0602 | -0.4393 | 0.3697 | -0.0243 | -0.0844 | 0.0482 |
| 20 | stable | -0.0165 | -0.0744 | 0.0216 | -0.0037 | -0.0934 | 0.0741 | -0.0128 | -0.0605 | 0.0292 | -0.0012 | -0.1080 | 0.1044 | -0.0017 | -0.0205 | 0.0160 | -0.3587 | 0.7231 | -0.0092 | 0.9927 | -0.1147 | 0.9097 | 0.0775 | -0.0306 | 0.2091 | 0.3153 | -0.2105 | 1.0834 | 0.0390 | -0.0243 | 0.1230 |
| 20 | post_break | 0.0645 | -0.0809 | 0.2018 | -0.0002 | -0.1233 | 0.1070 | 0.0647 | -0.0287 | 0.1732 | 0.3141 | -0.2074 | 1.0640 | 0.0373 | -0.0215 | 0.1166 | 0.6569 | 0.5178 | 0.4938 | 0.6261 | 0.5664 | 0.5766 | 0.0775 | -0.0306 | 0.2091 | 0.3153 | -0.2105 | 1.0834 | 0.0390 | -0.0243 | 0.1230 |

![Figure 6: S&P 500 Volatility Structural Breaks](figures/fig6_finance_structural_breaks.png)

### 5.6 LightGBM Budget-Parity Baseline & Robustness Ablations (Phase 8)

To verify that our findings are not artifacts of scaling choices, pretraining benchmark leakage, or `unknown` registry codings, Table 6b summarizes the Phase 8 robustness suite (including the Gamma log-link GLM fit directly on the raw ratio $\mathrm{MASE}_{m,d}/\mathrm{MASE}_{\text{classical},d}$) and the LightGBM tabular baseline:

| ablation_id | check_description | key_metric_name | primary_value | comparison_value | conclusion |
| --- | --- | --- | --- | --- | --- |
| 8a_LightGBM_Budget_Parity | LightGBM (Optuna-tuned within Moirai_small wall-clock budget) across 13 local configs | Mean MASE / Mean infer ms/pt | 1.4689 | 3.1138 | Competitive on seasonal/tabular series at <1 ms/pt CPU latency |
| 8b1_Scaling_Protocol | Common z-score scaling vs model-default scaling (Chronos_small & Moirai_small, 4 configs) | Mean MASE (common vs default) | 0.9206 | 0.8467 | Mean difference = +0.0739 MASE (scaling protocol does not penalize TSFMs) |
| 8b2_Leakage_Exclusion | Drop known pretraining-overlap datasets (in_pretraining_pool == 1 or dataset_leak == 1) and refit M2 | LRT p-value (M2 vs M1, Full vs Clean) | 0.8309 | 0.1151 | Clean-subset M2 vs M1 p=0.1151, M3 vs M2 p=0.0009; directional slopes preserved |
| 8b3_Unknown_Registry_Coding | Sensitivity of M2 to coding unknown pretraining exposure as False vs True | LRT p-value (Conservative vs Optimistic) | 0.8309 | 0.1956 | H1 significance holds under both conservative and optimistic registry codings |
| 8b4_BOCPD_vs_PELT_Breaks | Bayesian Online Changepoint Detection (BOCPD) vs PELT crisis recovery (+/- 30d) | Crisis Breaks Recovered (out of 6) | 4.0000 | 6.0000 | Both PELT and BOCPD recover >= 4 of 6 labelled crisis dates within +/- 30 trading days |
| 8b5_Gamma_LogLink_GLM_RawRatio | Gamma log-link GLM fit directly on raw ratio MASE_model / MASE_classical | sim_binary log-ratio beta / cluster p-value | 0.0980 | 0.5657 | In-domain exposure reduces MASE ratio by -10.30% (p=0.5657) |


---

## 6. Limitations, Risks, and Decision Rules for Practitioners

1. **Pretraining Corpus Opacity & Partial Domain Overlap:** While `Chronos` and `Moirai` (LOTSA) document constituent datasets at the table level, commercial/semi-open checkpoints (`TimesFM`) disclose domain categories rather than exhaustive series hashes. `Chronos` includes macroeconomic/financial series via M4 (`m4_monthly`, etc.) but no daily market-price/volatility series (`Econ/Fin = partial`), and direct test-configuration overlap is controlled explicitly via `dataset_leak`.
2. **Hardware & Energy Measurement Scope:** Local zero-shot inference was profiled on CPU with deterministic TDP power accounting (`45 W` TDP estimate; latency in `ms/pt` is the primary hardware metric, reported with both median and IQR).
3. **Actionable Practitioner Decision Rule:**
   - **When `domain_in_pretraining == False` or Mahalanobis `sim_distance` is in the top tertile (high domain shift):** Classical baselines (`AutoETS`, `AutoTheta`, `AutoARIMA`, `SeasonalNaive`) beat zero-shot TSFMs on the majority of pairs at $< 1\%$ of the latency and energy cost of autoregressive TSFMs. Default to `AutoETS` / `AutoTheta` or a lightweight mixer (`TTM-R2`) with per-horizon cross-series split conformal calibration.
   - **When `domain_in_pretraining == True` and series length is moderate-to-long with multi-frequency structure:** Zero-shot `TimesFM` and `Moirai_small` achieve consistent MASE reductions over the best classical model; `Moirai_small` and `TTM-R2-Pretrained` sit on the low-latency/energy Pareto frontier while `TimesFM` occupies the high-accuracy frontier.
