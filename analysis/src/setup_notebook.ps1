$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/4] Ensuring Python 3.13.7 is available via uv..."
uv python install 3.13.7

Write-Host "[2/4] Creating virtual environment in analysis/src/.venv..."
uv venv --python 3.13.7 .venv

Write-Host "[3/4] Installing pinned notebook dependencies..."
uv pip install --python .\.venv\Scripts\python.exe -r requirements-pinned.txt

Write-Host "[4/4] Registering Jupyter kernel for VS Code..."
.\.venv\Scripts\python.exe -m ipykernel install --user --name plopp-notebook --display-name "PLOPP Notebook (.venv)"

Write-Host "Notebook environment is ready. Open analysis/src/spectra_analysis.ipynb and select kernel: PLOPP Notebook (.venv)."