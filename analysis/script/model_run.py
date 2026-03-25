"""
model_run.py
============
Pipeline construction, training, and prediction for the PLOPP/FLOPP spectral
classification pipeline.

The full sklearn Pipeline is:

    SNVTransformer  →  DerivativeTransformer  →  PCA  →  RandomForestClassifier

A custom probability threshold (default 0.60) is applied at prediction time so
that a spectrum is labelled PLoPP only when P(PLoPP) ≥ threshold.

Usage
-----
    from model_run import build_pipeline, train_pipeline, predict, save_pipeline, load_pipeline

    pipeline = build_pipeline()
    pipeline = train_pipeline(pipeline, X_train, y_train)

    y_pred, y_proba = predict(pipeline, X_test)

    save_pipeline(pipeline, path="models/rf_pipeline.pkl")
    pipeline = load_pipeline("models/rf_pipeline.pkl")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config
from data_prep import SNVTransformer, DerivativeTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(
    snv_use_scaling: bool = config.SNV_USE_SCALING,
    derivative_window: int = config.DERIVATIVE_WINDOW,
    derivative_polyorder: int = config.DERIVATIVE_POLYORDER,
    derivative_order: int = config.DERIVATIVE_ORDER,
    pca_variance: float = config.PCA_VARIANCE_RETAINED,
    n_estimators: int = config.N_ESTIMATORS,
    max_depth: int | None = config.MAX_DEPTH,
    max_features: str | int | float | None = config.MAX_FEATURES,
    random_state: int = config.RANDOM_STATE,
) -> Pipeline:
    """
    Construct the full preprocessing + classification pipeline.

    Parameters
    ----------
    snv_use_scaling : bool
        Whether to scale by standard deviation in the SNV step.
    derivative_window : int
        Savitzky-Golay window length for derivative computation.
    derivative_polyorder : int
        Polynomial order for the Savitzky-Golay filter.
    derivative_order : int
        Order of derivative (1 = first derivative).
    pca_variance : float
        Fraction of variance to retain in the PCA step (0 < pca_variance ≤ 1).
    n_estimators : int
        Number of trees in the random forest.
    max_depth : int or None
        Maximum tree depth; None lets trees grow fully.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline(
        steps=[
            (
                "snv",
                SNVTransformer(use_scaling=snv_use_scaling),
            ),
            (
                "derivative",
                DerivativeTransformer(
                    window_length=derivative_window,
                    polyorder=derivative_polyorder,
                    deriv=derivative_order,
                ),
            ),
            (
                "pca",
                PCA(n_components=pca_variance, random_state=random_state),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    random_state=random_state,
                ),
            ),
        ]
    )
    logger.info("Pipeline constructed: %s", " → ".join(name for name, _ in pipeline.steps))
    return pipeline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fit *pipeline* on the provided training data.

    Parameters
    ----------
    pipeline : Pipeline
        An unfitted sklearn Pipeline (e.g. from build_pipeline()).
    X_train : pd.DataFrame
        Training feature matrix (wavenumber columns).
    y_train : pd.Series
        Training class labels.

    Returns
    -------
    Pipeline
        The same pipeline object, now fitted.
    """
    logger.info(
        "Training pipeline on %d samples with %d features …",
        len(X_train), X_train.shape[1],
    )
    pipeline.fit(X_train, y_train)
    n_components = pipeline.named_steps["pca"].n_components_
    logger.info("PCA retained %d components (%.0f%% variance)", n_components, config.PCA_VARIANCE_RETAINED * 100)
    logger.info("Training complete.")
    return pipeline


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = config.PARAM_GRID,
    cv: int = config.CV_FOLDS,
    scoring: str = config.CV_SCORING,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """
    Run GridSearchCV over the full pipeline to find optimal RF hyperparameters.

    Uses 5-fold cross-validation on the training set only.  Returns the best
    parameters (stripped of the 'classifier__' prefix) so they can be passed
    directly to build_pipeline() and saved to hyperparameters.json.

    Parameters
    ----------
    X_train     : training feature matrix
    y_train     : training labels
    param_grid  : dict of pipeline param names → values to search
    cv          : number of CV folds
    scoring     : sklearn scoring metric
    random_state: random seed passed to the base pipeline

    Returns
    -------
    dict  e.g. {"n_estimators": 200, "max_depth": None, "max_features": "log2"}
    """
    logger.info("Starting %d-fold GridSearchCV over %d combinations …",
                cv, len(list(__import__("itertools").product(*param_grid.values()))))

    base_pipeline = build_pipeline(random_state=random_state)
    search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=False,  # we'll rebuild with best params ourselves
    )
    search.fit(X_train, y_train)

    best_raw = search.best_params_
    best_score = search.best_score_

    # Strip "classifier__" prefix so params map directly to build_pipeline kwargs
    best_params = {k.replace("classifier__", ""): v for k, v in best_raw.items()}

    logger.info("Best CV score (%s): %.4f", scoring, best_score)
    logger.info("Best params: %s", best_params)
    return best_params


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_param_grids: dict = None,
    cv: int = config.CV_FOLDS,
    scoring: str = config.CV_SCORING,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """
    GridSearch over multiple classifiers and return the best params overall.

    Matches notebook cell 12: tests Random Forest, Logistic Regression, SVM,
    and XGBoost, each with their own param grid, and selects the winner by
    best CV F1 score.  Results are saved to models/best_params.json.

    Parameters
    ----------
    X_train          : training feature matrix
    y_train          : training labels (original string labels e.g. 'PLoPP'/'FLoPP')
    model_param_grids: dict of {model_name: param_grid}; defaults to
                       config.ALL_MODELS_PARAM_GRIDS
    cv               : number of CV folds
    scoring          : sklearn scoring string (notebook uses 'f1')
    random_state     : random seed

    Returns
    -------
    dict with keys:
        "best_model_name"  : str   – winning model name
        "best_f1_score"    : float – best CV F1
        "best_params"      : dict  – raw best_params_ from GridSearchCV
    """
    if model_param_grids is None:
        model_param_grids = config.ALL_MODELS_PARAM_GRIDS

    _model_instances = {
        "Random Forest":     RandomForestClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM":               SVC(probability=True, random_state=random_state),
        "XGBoost":           XGBClassifier(random_state=random_state, verbosity=0),
    }

    best_model_name = None
    best_f1_score = 0.0
    best_params: dict = {}

    for name, param_grid in model_param_grids.items():
        model = _model_instances.get(name)
        if model is None:
            logger.warning("No model instance defined for '%s' — skipping.", name)
            continue

        pipeline = Pipeline([
            ("snv",         SNVTransformer()),
            ("derivatives", DerivativeTransformer()),
            ("classifier",  model),
        ])

        logger.info("GridSearchCV for %s …", name)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, verbose=1)
        grid_search.fit(X_train, y_train)

        logger.info("Best params for %s: %s", name, grid_search.best_params_)
        logger.info("Best CV F1 for %s: %.3f", name, grid_search.best_score_)

        if grid_search.best_score_ > best_f1_score:
            best_f1_score = grid_search.best_score_
            best_model_name = name
            best_params = grid_search.best_params_

    logger.info(
        "Winner: %s with CV F1 = %.3f  params = %s",
        best_model_name, best_f1_score, best_params,
    )

    config.save_hyperparameters({"best_model": best_model_name, **best_params})

    return {
        "best_model_name": best_model_name,
        "best_f1_score":   best_f1_score,
        "best_params":     best_params,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    pipeline: Pipeline,
    X: pd.DataFrame,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate class predictions and class probabilities.

    A custom probability threshold is applied: a sample is predicted as
    PLoPP when P(PLoPP) ≥ *threshold*, otherwise FLoPP.

    Parameters
    ----------
    pipeline  : Pipeline   – fitted pipeline
    X         : pd.DataFrame – feature matrix to predict on
    threshold : float      – minimum P(PLoPP) required to assign PLoPP label

    Returns
    -------
    y_pred  : np.ndarray, shape (n_samples,)   – predicted class labels
    y_proba : np.ndarray, shape (n_samples, 2) – class probabilities
                column 0 = P(FLoPP), column 1 = P(PLoPP)
    """
    y_proba = pipeline.predict_proba(X)
    classes = pipeline.classes_

    # Locate the PLoPP column index
    plopp_idx = list(classes).index(config.LABEL_PLOPP)

    y_pred = np.where(
        y_proba[:, plopp_idx] >= threshold,
        config.LABEL_PLOPP,
        config.LABEL_FLOPP,
    )
    return y_pred, y_proba


# ---------------------------------------------------------------------------
# Sector binary classifiers  (matches notebook Cell 49)
# ---------------------------------------------------------------------------

def train_sector_binary_models(
    all_with_sectors: pd.DataFrame,
    sectors: list | None = None,
    test_size: float = 0.3,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """
    Train one binary classifier per sector.

    Matches notebook Cell 46 exactly:
    - Input is ALL data (PLOPP + FLOPP) merged with sector map, as returned by
      data_prep.load_all_with_sectors().  FLOPP rows have NaN for Sector and
      become class 0 ("not this sector"), exactly as the notebook does.
    - Binary labels: 1 = target sector, 0 = everything else (including FLOPP
      and PLOPP from other sectors).
    - Drops the same columns as the corrected notebook sector path:
        ['Unnamed: 0', 'Sector', 'Target', 'Sample', 'Sample_x', 'Sample_y',
         'Sample_Number', 'Color', 'CSVName', 'SectorBinary']  (errors='ignore')
    - Pipeline: SNVTransformer(use_scaling=True) → DerivativeTransformer(deriv=1)
                → PCA(n_components=0.95) → RandomForestClassifier(random_state=42)
    - Split: test_size=0.3, random_state=42

    Parameters
    ----------
    all_with_sectors : DataFrame from data_prep.load_all_with_sectors()
    sectors          : list of sector names to model; defaults to config.SECTORS
    test_size        : fraction held out (notebook uses 0.3)
    random_state     : random seed (notebook uses 42)

    Returns
    -------
    dict  {sector_name: {"pipeline": ..., "y_test": ..., "y_pred": ...}}
    """
    if sectors is None:
        sectors = config.SECTORS

    # Matches notebook Cell 46 drop list (errors='ignore' handles absent columns)
    _drop_cols = [
        "Unnamed: 0", "Sector", "Target", "Sample", "Sample_x", "Sample_y",
        "Sample_Number", "Color", "CSVName", "SectorBinary",
    ]

    results = {}
    for sector in sectors:
        data = all_with_sectors.copy()
        # NaN Sector rows (FLOPP + unmatched PLOPP) evaluate False → SectorBinary=0
        data["SectorBinary"] = (data["Sector"] == sector).astype(int)

        X = data.drop(columns=_drop_cols, errors="ignore")
        X.columns = X.columns.astype(str)
        y = data["SectorBinary"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Explicit pipeline params matching notebook Cell 46
        pipe = Pipeline([
            ("snv",        SNVTransformer(use_scaling=True)),
            ("derivative", DerivativeTransformer(deriv=1)),
            ("pca",        PCA(n_components=config.PCA_VARIANCE_RETAINED)),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        n_components = pipe.named_steps["pca"].n_components_
        logger.info(
            "Sector '%s': trained on %d samples, tested on %d, PCA=%d components",
            sector, len(X_train), len(X_test), n_components,
        )
        results[sector] = {
            "pipeline": pipe,
            "y_test":   y_test.reset_index(drop=True),
            "y_pred":   pd.Series(y_pred, name="predicted"),
            "X_test":   X_test.reset_index(drop=True),
            "train_columns": list(X_train.columns),
        }

    return results


def evaluate_sector_models_on_validation(
    sector_models: dict,
    X_val: pd.DataFrame,
    meta: pd.DataFrame,
    sector_col: str = "zoie_sector",
    sectors: list | None = None,
) -> dict:
    """
    Evaluate each trained sector binary model on external validation data.

    Matches notebook Cell 57: for each sector, labels validation samples
    as 1 (= that sector) or 0 (= other), then predicts with the trained
    sector pipeline.

    Parameters
    ----------
    sector_models : dict returned by train_sector_binary_models()
    X_val         : validation feature matrix
    meta          : validation metadata with sector_col column
    sector_col    : column name holding sector labels in meta
    sectors       : list of sector names to evaluate; defaults to all keys in
                    sector_models. Pass config.VALIDATION_SECTORS to mirror
                    the notebook's explicit target_sectors loop.

    Returns
    -------
    dict  {sector_name: {"y_test": ..., "y_pred": ...}}
    """
    if sectors is None:
        sectors = list(sector_models.keys())

    results = {}
    for sector in sectors:
        if sector not in sector_models:
            logger.warning("Sector '%s' not found in trained models — skipping.", sector)
            continue

        model_data = sector_models[sector]
        pipe = model_data["pipeline"]

        train_columns = model_data.get("train_columns")
        if train_columns is None and hasattr(pipe, "feature_names_in_"):
            train_columns = list(pipe.feature_names_in_)

        # Align X_val columns to what this pipeline was trained on.
        # This is required for notebook parity because the sector models can
        # include columns such as `Unnamed: 0` from merged_df.csv.
        if train_columns is not None:
            X_for_pred = X_val.copy()
            X_for_pred.columns = X_for_pred.columns.astype(str)
            X_for_pred = X_for_pred.reindex(columns=train_columns, fill_value=0)
        else:
            X_for_pred = X_val

        y_true = (meta[sector_col] == sector).astype(int).reset_index(drop=True)
        y_pred = pd.Series(pipe.predict(X_for_pred), name="predicted")

        logger.info(
            "Sector '%s' validation: %d samples (pos=%d)",
            sector, len(y_true), y_true.sum(),
        )
        results[sector] = {
            "y_test": y_true,
            "y_pred": y_pred,
        }
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    """
    Serialise *pipeline* to *path* using pickle.

    Parameters
    ----------
    pipeline : Pipeline  – fitted pipeline to save
    path     : str | Path – destination file path (.pkl recommended)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(pipeline, fh)
    logger.info("Pipeline saved → %s", path)


def load_pipeline(path: str | Path) -> Pipeline:
    """
    Load a previously saved pipeline from *path*.

    Parameters
    ----------
    path : str | Path – path to the pickle file

    Returns
    -------
    Pipeline
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved pipeline found at: {path}")
    with open(path, "rb") as fh:
        pipeline = pickle.load(fh)
    logger.info("Pipeline loaded ← %s", path)
    return pipeline
