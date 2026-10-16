# Domain-Shift and Computational-Efficiency Benchmark of Time-Series Foundation Models

This repository implements an end-to-end empirical benchmark and statistical modeling pipeline evaluating:
1. **Domain-Shift Survival & Predictability (H1–H3):** Whether zero-shot Time-Series Foundation Model (TSFM) gains over classical baselines (`SeasonalNaive`, `AutoETS`, `AutoTheta`, `AutoARIMA`) are predicted by **domain similarity to documented pretraining corpora** (`domain_in_pretraining`, continuous Mahalanobis/k-NN `sim_distance`) beyond intrinsic series difficulty (`forecastability`, `trend_strength`, `seasonal_strength`, `hurst`, `stability`, `lumpiness`, `log_length`).
2. **Compute-Efficiency & Pareto Optimality (H4):** Full runtime, peak memory (RSS/VRAM), and energy (`kWh`) accounting across models and datasets, mapping the region of feature space where classical baselines are Pareto-optimal.

## Quickstart

```powershell
# 1. Create Python 3.11 environment and install package
.\make.ps1 env

# 2. Run Phase 0 regression check and unit tests
.\make.ps1 phase0
.\make.ps1 test

# 3. Run individual phases or the complete pipeline
.\make.ps1 data
.\make.ps1 baselines
.\make.ps1 features
.\make.ps1 models
.\make.ps1 uncertainty
.\make.ps1 mixed-models
.\make.ps1 finance
.\make.ps1 ablations
.\make.ps1 report
```

On POSIX systems with GNU `make`:
```bash
make env && make test && make all
```

## Repository Structure

- `configs/` — Schema-validated YAML configs (`datasets.yaml`, `models.yaml`, `pretraining_registry.yaml`, `experiment.yaml`)
- `envs/` — Per-environment pinned requirement files (`core.txt`, `chronos.txt`, `moirai.txt`, `timesfm.txt`, `ttm.txt`)
- `seed/baseline_model.py` — Reference AirPassengers baseline harness (Section 3 of Milestone 1 report)
- `src/tsdomain/` — Core package (`data/`, `baselines/`, `models/`, `features/`, `metrics/`, `compute/`, `analysis/`)
- `results/` — Generated forecasts (`parquet` + `.meta.json`), metrics, compute profiles, features, fitted models, figures, and tables
- `docs/` — `decisions.md`, `walkthrough.md`, `open_questions.md`, `literature_review.md`
- `report/` — Final reproducible research report assembled strictly from `results/`
