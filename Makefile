.PHONY: all env test phase0 data baselines features models uncertainty mixed-models finance ablations report clean

PYTHON ?= uv run python
PYTEST ?= uv run pytest

all: test data baselines features models uncertainty mixed-models finance ablations report

env:
	uv venv .venv --python 3.11
	uv pip install -e .

test:
	$(PYTEST) tests/

phase0:
	$(PYTHON) seed/baseline_model.py
	$(PYTEST) tests/test_metrics.py tests/test_phase0_airpassengers.py

data:
	$(PYTHON) -m tsdomain.cli data

baselines:
	$(PYTHON) -m tsdomain.cli baselines

features:
	$(PYTHON) -m tsdomain.cli features

models:
	$(PYTHON) -m tsdomain.cli models

uncertainty:
	$(PYTHON) -m tsdomain.cli uncertainty

mixed-models:
	$(PYTHON) -m tsdomain.cli mixed-models

finance:
	$(PYTHON) -m tsdomain.cli finance

ablations:
	$(PYTHON) -m tsdomain.cli ablations

report:
	$(PYTHON) -m tsdomain.cli report
