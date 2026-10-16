# Open Questions & Manual Owner Verification (`docs/open_questions.md`)

In accordance with Section 7 of the Build Plan, this file tracks all claims, licences, and corpus details that must be verified by hand by the project owner and are never guessed silently.

## 1. Pretraining Registry Claims Marked `verified: false` (`configs/pretraining_registry.yaml`)
- **Moirai 1.1 / LOTSA Economics & Finance Sub-Datasets:**
  - *Status:* `verified: false`
  - *Context:* LOTSA (Woo et al., 2024) documents an `Econ/Fin` domain category (including datasets such as `FRED`, `Monash Macro`, and `Bitcoin/Crypto`), but whether any equity index volatility or S&P 500 OHLC series overlap post-1990 daily volatility windows requires manual verification against Appendix C of the Moirai paper.
- **TimesFM (Google Research) Real-World Pretraining Mix:**
  - *Status:* `verified: false`
  - *Context:* Das et al. (2024) document pretraining on Google Trends, Wiki Pageviews, M4, Electricity, Traffic, and Weather, plus synthetic series. Exact inclusion/exclusion of financial market return/volatility series in the internal Google Trends queries or commercial checkpoints (`timesfm-1.0-200m` / `timesfm-2.0-500m`) is not fully enumerated at the ticker level.
- **IBM Granite Tiny Time Mixers (TTM-R2) Pretraining Cutoff & Domain Mix:**
  - *Status:* `verified: false`
  - *Context:* Ekambaram et al. (2024) train TTM on a non-leaking subset of Monash/LotSA datasets (~244M–1B samples). Exact cutoff timestamps per sub-dataset should be cross-checked against the Hugging Face model card (`ibm-granite/granite-timeseries-ttm-r2`).

## 2. Unseen-Domain Dataset Licences & Corpus Absence
- **Unseen Dataset 1 (`unseen_mandi` — Indian Wholesale Agricultural Commodity Prices from Agmarknet):**
  - *Source:* Directorate of Marketing & Inspection (DMI), Ministry of Agriculture and Farmers Welfare, Government of India (`agmarknet.gov.in`), accessed via the public archival mirror `https://github.com/iancovert/Agmarknet`.
  - *Verification Needed:* Confirm coursework redistribution terms under the Government of India Open Data License (GODL-India) and verify that Agmarknet mandi price series are absent from `LOTSA`, `Chronos`, and `TimesFM` pretraining corpora (corroborated by `Viharikvs/TFSMAG` dataset card, which explicitly notes that LOTSA, Chronos, and TimesFM contain zero agricultural mandi data).
- **Unseen Dataset 2 (`unseen_recency` — Post-Cutoff 2025–2026 Daily Weather Series via Open-Meteo):**
  - *Source:* Open-Meteo Historical Weather API (`https://archive-api.open-meteo.com/v1/archive`, ERA5 reanalysis).
  - *Verification Needed:* Open-Meteo data is licensed under Attribution 4.0 International (CC BY 4.0) for non-commercial/academic use; owner should confirm attribution string in the final report submission.

## 3. GIFT-Eval Leaderboard Leakage Flag Semantics
- *Observation:* In `SalesforceAIResearch/gift-eval`, each model's `config.json` includes `"testdata_leakage": "Yes" | "No"`, and a separate `model_type` (`"zero-shot"`, `"pretrained"`, `"statistical"`, `"deep-learning"`, `"agentic"`). Per the 2025-08-25 GIFT-Eval update log, `"testdata_leakage": "No"` indicates test split timestamps did not overlap pretraining, while `"model_type": "zero-shot"` vs `"pretrained"` distinguishes whether GIFT-Eval training splits were used.
- *Owner Check:* Confirm whether Phase 8's leakage exclusion ablation should filter on `test_data_leakage_flag == True` only, or also exclude `model_type == "pretrained"` models that touched GIFT-Eval training splits. (Both variants are implemented and reported in Phase 8.)
