# PLOPP/FLOPP Spectral Classification

This folder contains the script-based version of the FTIR paint / non-paint
analysis that was originally developed in
`analysis/src/spectra_analysis.ipynb`.

The goal of this code is to make the notebook workflow reproducible and easier
to run as a normal Python pipeline while preserving the notebook's modeling and
validation logic as closely as practical.

## Layout

```text
analysis/
├── data/                input datasets used by the script
├── figures/             generated figures
├── models/              saved model artifacts
├── src/                 original notebook reference
└── script/              this script-based pipeline
    ├── config.py
    ├── data_prep.py
    ├── evaluation.py
    ├── main.py
    ├── model_run.py
    ├── pyproject.toml
    └── uv.lock
```

## Environment

This project is managed with `uv` and expects Python `>=3.11`.

From `analysis/script`:

```bash
uv sync
```

If you already have a virtual environment active from somewhere else, `uv` may
warn that it is ignoring it. That is expected unless you intentionally use
`uv run --active ...`.

## Running

From `analysis/script`:

```bash
uv run python main.py
```

Common options:

```bash
uv run python main.py --skip-tuning
uv run python main.py --log-level DEBUG
uv run python main.py --log-level WARNING
```

## What The Pipeline Does

1. Loads the PLOPP and FLOPP spectra.
2. Builds the train / test split.
3. Fits the SNV -> derivative -> PCA -> RandomForest pipeline.
4. Evaluates the held-out test set.
5. Evaluates the external validation data.
6. Trains and evaluates the sector models.
7. Writes figures, logs, and model artifacts under `analysis/`.

## Main Modules

| File | Purpose |
|---|---|
| `config.py` | Paths, constants, and model settings |
| `data_prep.py` | Data loading, preprocessing, and notebook-parity validation assembly |
| `model_run.py` | Training, prediction, sector-model training, and persistence helpers |
| `evaluation.py` | Metrics, reports, and figure generation |
| `main.py` | End-to-end entry point |

## Outputs

Typical outputs are written under `analysis/`:

- `figures/`
- `models/`
- `pipeline.log`
- `validation_results.csv`

Exact figure filenames depend on which stages are run.

## Notes

- The original notebook is still the reference for exploratory work.
- The script currently tracks the corrected sector-validation behavior rather
  than the stale figures embedded in older notebook outputs.
- If notebook and script results differ, restart the notebook kernel and rerun
  the relevant cells in order before comparing outputs.
