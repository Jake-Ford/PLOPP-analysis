"""
config.py
=========
Central configuration for the PLOPP/FLOPP spectral classification pipeline.
All file paths, hyperparameters, and constants are defined here so they never
need to be hunted down inside processing or model code.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths (all relative to this file's location inside analysis/src_v2/)
# ---------------------------------------------------------------------------
SRC_DIR   = Path(__file__).resolve().parent
ANALYSIS_DIR = SRC_DIR.parent                          # analysis/
PROJECT_DIR  = ANALYSIS_DIR.parent                     # PLOPP-analysis/

# Raw spectral data (lives under analysis/data/)
DATA_DIR          = ANALYSIS_DIR / "data"
PLOPP_DATA_DIR    = DATA_DIR / "plopp"
FLOPP_DATA_DIR    = DATA_DIR / "flopp"
ANDREW_TURNER_DIR = DATA_DIR / "andrew_turner"
CITADEL_DATA_DIR  = DATA_DIR / "citadel"
CITADEL_SAMPLE_KEY = CITADEL_DATA_DIR / "2024.11.27_Citadel_sampleKey_ZoieForJake.xlsx"
SECTOR_MAP_PATH    = DATA_DIR / "sector_map.csv"
NOTEBOOK_MERGED_DF_PATH = PROJECT_DIR / "data" / "output" / "merged_df.csv"
NOTEBOOK_SECTOR_COLOR_PATH = PROJECT_DIR / "data" / "merged_df_sector_color.csv"

# Output directories
MODELS_DIR  = ANALYSIS_DIR / "models"
FIGURES_DIR = ANALYSIS_DIR / "figures"

# ---------------------------------------------------------------------------
# Data-split settings
# ---------------------------------------------------------------------------
TEST_SIZE    = 0.4
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Preprocessing settings
# ---------------------------------------------------------------------------
SNV_USE_SCALING       = True
DERIVATIVE_WINDOW     = 11
DERIVATIVE_POLYORDER  = 2
DERIVATIVE_ORDER      = 1        # 1st derivative
PCA_VARIANCE_RETAINED = 0.95     # retain 95 % of variance

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
N_ESTIMATORS  = 100
MAX_DEPTH     = None             # let trees grow fully (RF default)
MAX_FEATURES  = "log2"           # features considered at each split
CLASSIFICATION_THRESHOLD = 0.60  # P(PLoPP) >= 0.60 → predicted PLoPP (matches notebook)

# Hyperparameter tuning (GridSearchCV, 5-fold CV)
CV_FOLDS = 5
CV_SCORING = "f1_weighted"
PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth":    [None, 3, 5, 10, 25],
    "classifier__max_features": ["sqrt", "log2"],
}

# Multi-model sweep (matches notebook cell 12)
# Base param grid applied to every model; each model dict adds classifier-specific entries.
ALL_MODELS_BASE_GRID = {
    "snv__use_scaling":    [True, False],
    "derivatives__deriv":  [0, 1, 2],
}

ALL_MODELS_PARAM_GRIDS = {
    "Random Forest": {
        **ALL_MODELS_BASE_GRID,
        "classifier__n_estimators":      [100, 200, 300, 500],
        "classifier__max_depth":         [3, 6, 9],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf":  [1, 2, 4],
        # Notebook uses ['auto','sqrt','log2'] — 'auto' removed in sklearn>=1.4 (was alias
        # for 'sqrt'); replaced with None (use all features) to keep 3 values → 1944 candidates
        "classifier__max_features":      ["sqrt", "log2", None],
    },
    "Logistic Regression": {
        **ALL_MODELS_BASE_GRID,
        "classifier__penalty": ["l1", "l2", "elasticnet", None],
        "classifier__C":       [0.01, 0.1, 1, 10, 100],
        "classifier__solver":  ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    },
    "SVM": {
        **ALL_MODELS_BASE_GRID,
        "classifier__C":      [0.1, 1, 10, 100],
        "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "classifier__gamma":  ["scale", "auto"],
    },
    "XGBoost": {
        **ALL_MODELS_BASE_GRID,
        "classifier__n_estimators":  [100, 500],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth":     [3, 6, 9],
    },
}

# Class labels
LABEL_PLOPP = "PLoPP"
LABEL_FLOPP = "FLoPP"

# Sectors analysed in the binary sector classifiers (matches notebook Cell 49)
SECTORS = [
    "Automotive",
    "General Industrial",
    "Architectural",
    "Road Marking",
    "Consumer",
    "Wood",
    "Marine",
]

# Sectors evaluated on external validation data.
# The notebook loops over all trained target_sectors; Automotive and Consumer
# simply have zero positive examples in the external validation set.
VALIDATION_SECTORS = [
    "Automotive",
    "General Industrial",
    "Architectural",
    "Road Marking",
    "Consumer",
    "Marine",
    "Wood",
]

# ---------------------------------------------------------------------------
# Figure settings
# ---------------------------------------------------------------------------
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def resolve_data_path(relative_path: Path) -> Path:
    """Resolve a data path relative to src_v2/ and return an absolute Path."""
    return (SRC_DIR / relative_path).resolve()


def ensure_output_dirs() -> None:
    """Create output directories if they do not already exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_hyperparameters(
    tuned_params: dict | None = None,
    path: Path | None = None,
) -> None:
    """
    Write model hyperparameters to a JSON file for reproducibility.

    If *tuned_params* is provided (from GridSearchCV) those values are used;
    otherwise the config defaults are written.
    """
    if path is None:
        path = MODELS_DIR / "hyperparameters.json"
    defaults = {
        "snv":                      SNV_USE_SCALING,
        "derivative_order":         DERIVATIVE_ORDER,
        "n_estimators":             N_ESTIMATORS,
        "max_depth":                MAX_DEPTH,
        "max_features":             MAX_FEATURES,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "random_state":             RANDOM_STATE,
    }
    params = {**defaults, **(tuned_params or {})}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)

