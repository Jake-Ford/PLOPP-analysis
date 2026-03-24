# PLOPP/FLOPP Spectral Classification — src_v2

Refactored, production-quality Python implementation of the PLOPP/FLOPP
FTIR spectral classification pipeline.  The original exploratory notebook
(`../src/spectra_analysis.ipynb`) has been reorganised into four focused
modules with clean interfaces and no notebook clutter.

---

## Repository layout

```
analysis/
├── figures/                 ← generated figures (created on first run)
├── models/                  ← saved pipeline artifacts
│   └── rf_pipeline.pkl
├── src/                     ← original notebook (reference)
└── src_v2/                  ← this package
    ├── config.py            ← all paths, constants, hyperparameters
    ├── data_prep.py         ← data loading, cleaning, transformers, splitting
    ├── model_run.py         ← pipeline construction, training, prediction, I/O
    ├── evaluation.py        ← metrics computation and figure generation
    ├── main.py              ← end-to-end orchestration entry point
    └── requirements.txt     ← pinned Python dependencies
```

---

## Environment setup

The project virtual environment lives one level above this folder.
Activate it before running anything:

```bash
# From analysis/src_v2/
source ../../venv/bin/activate
```

To create the environment from scratch:

```bash
cd /path/to/PLOPP-analysis
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r analysis/src_v2/requirements.txt
```

---

## Running the pipeline

```bash
# Activate environment first (see above), then:
cd analysis/src_v2
python main.py
```

Optional flags:

```bash
python main.py --log-level DEBUG   # verbose output
python main.py --log-level WARNING # quiet output
```

The pipeline will:
1. Load PLOPP and FLOPP spectral CSVs, split into train/test sets
2. Fit the SNV → Derivative → PCA → RandomForest pipeline
3. Evaluate on the held-out test set and write figures to `../figures/`
4. Evaluate on Andrew Turner + Citadel validation sets
5. Save the fitted pipeline to `../models/rf_pipeline.pkl`

---

## Module overview

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for all paths and hyperparameters |
| `data_prep.py` | CSV loading, label assignment, group-aware splitting, SNVTransformer, DerivativeTransformer, validation loaders |
| `model_run.py` | `build_pipeline()`, `train_pipeline()`, `predict()` (with threshold), `save_pipeline()`, `load_pipeline()` |
| `evaluation.py` | `compute_metrics()`, confusion matrix, ROC curve, feature importance plots, probability distribution plot, `evaluate_validation_set()` |
| `main.py` | Glues everything together; run this file to execute the full pipeline |

---

## Pipeline architecture

```
Raw CSVs
  └─► _process_csv_directory()   (data_prep)
        └─► SNVTransformer         (mean-centre + scale row-wise)
              └─► DerivativeTransformer  (1st derivative, Savitzky-Golay)
                    └─► PCA              (retain 95 % variance)
                          └─► RandomForestClassifier
                                └─► threshold = 0.60 → PLoPP / FLoPP
```

---

## Output files

| File | Description |
|---|---|
| `../figures/confusion_matrix.png` | Annotated confusion matrix with TPR/FPR/TNR/FNR |
| `../figures/roc_curve.png` | ROC curve with AUC |
| `../figures/feature_importance_top15.png` | Top 15 wavenumber importances |
| `../figures/feature_importance_spectrum.png` | Full-spectrum importance line plot |
| `../figures/predicted_probabilities.png` | Scatter + box plot of P(PLoPP) |
| `../models/rf_pipeline.pkl` | Serialised fitted pipeline |
| `../validation_results.csv` | Per-sample validation predictions |
| `../pipeline.log` | Full run log |
