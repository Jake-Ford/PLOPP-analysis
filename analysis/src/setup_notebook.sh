#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/4] Ensuring Python 3.13.7 is available via uv..."
uv python install 3.13.7

echo "[2/4] Creating virtual environment in analysis/src/.venv..."
uv venv --python 3.13.7 .venv

echo "[3/4] Installing pinned notebook dependencies..."
uv pip install --python .venv/bin/python -r requirements-pinned.txt

echo "[4/4] Registering Jupyter kernel for VS Code..."
.venv/bin/python -m ipykernel install --user --name plopp-notebook --display-name "PLOPP Notebook (.venv)"

echo "Notebook environment is ready. Open analysis/src/spectra_analysis.ipynb and select kernel: PLOPP Notebook (.venv)."