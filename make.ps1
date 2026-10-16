param(
    [Parameter(Position=0)]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$UV = "C:\Users\rishi\.local\bin\uv.exe"
if (-not (Test-Path $UV)) { $UV = "uv" }
$PY = ".\.venv\Scripts\python.exe"
$PYTEST = ".\.venv\Scripts\pytest.exe"

switch ($Target) {
    "env" {
        & $UV venv .venv --python 3.11
        & $UV pip install -e .
    }
    "test" {
        & $PYTEST tests/
    }
    "phase0" {
        & $PY seed/baseline_model.py
        & $PYTEST tests/test_metrics.py tests/test_phase0_airpassengers.py
    }
    "data" { & $PY -m tsdomain.cli data }
    "baselines" { & $PY -m tsdomain.cli baselines }
    "features" { & $PY -m tsdomain.cli features }
    "models" { & $PY -m tsdomain.cli models }
    "uncertainty" { & $PY -m tsdomain.cli uncertainty }
    "mixed-models" { & $PY -m tsdomain.cli mixed-models }
    "finance" { & $PY -m tsdomain.cli finance }
    "ablations" { & $PY -m tsdomain.cli ablations }
    "report" { & $PY -m tsdomain.cli report }
    "all" {
        & $PYTEST tests/
        & $PY -m tsdomain.cli all
    }
    default {
        Write-Error "Unknown target: $Target"
    }
}
