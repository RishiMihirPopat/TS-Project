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
