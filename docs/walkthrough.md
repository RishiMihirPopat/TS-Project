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
