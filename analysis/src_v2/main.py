"""
main.py
=======
Entry point for the PLOPP/FLOPP spectral classification pipeline.

Execution order
---------------
1.  Load and prepare training data          (data_prep)
2.  Build and train the classification pipeline  (model_run)
3.  Evaluate on the held-out test set       (evaluation)
4.  Load and evaluate external validation sets  (data_prep + evaluation)
5.  Save the trained pipeline               (model_run)

Run
---
    # From analysis/src_v2/:
    python main.py

    # Or with explicit log level:
    python main.py --log-level DEBUG
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import config
import data_prep
import evaluation
import model_run


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.ANALYSIS_DIR / "pipeline.log", mode="a"),
        ],
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_load_training_data():
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("STEP 1 – Loading training data")
    logger.info("=" * 60)
    X_train, y_train, X_test, y_test, test_samples = data_prep.load_training_data()
    print("Test set shape:", X_test.shape)
    print("Test label counts:\n", y_test.value_counts().to_string())
    logger.info(
        "Train class counts:\n%s",
        y_train.value_counts().to_string(),
    )
    logger.info(
        "Test class counts:\n%s",
        y_test.value_counts().to_string(),
    )
    return X_train, y_train, X_test, y_test, test_samples


def step_tune_pipeline(X_train, y_train, skip_tuning: bool = False):
    logger = logging.getLogger("main")
    logger.info("=" * 60)

    if skip_tuning:
        path = config.MODELS_DIR / "best_params.json"
        if not path.exists():
            raise FileNotFoundError(
                f"--skip-tuning requested but no best_params.json found at {path}. "
                "Run without --skip-tuning first to generate it."
            )
        with open(path) as f:
            params = json.load(f)
        logger.info("STEP 2 – Skipping tuning, loaded params from %s", path)
        logger.info("Loaded params: %s", params)
        return params

    logger.info("STEP 2 – Hyperparameter tuning: multi-model sweep (%d-fold CV)", config.CV_FOLDS)
    logger.info("=" * 60)
    result = model_run.tune_all_models(X_train, y_train)
    logger.info(
        "Best model: %s  |  CV F1: %.3f",
        result["best_model_name"], result["best_f1_score"],
    )
    logger.info("Tuned hyperparameters saved.")
    return result["best_params"]


def step_train_pipeline(X_train, y_train, best_params: dict):
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("STEP 3 – Building and training pipeline with tuned params")
    logger.info("=" * 60)
    # best_params from tune_all_models has 'classifier__' prefixes — strip them
    stripped = {k.replace("classifier__", ""): v for k, v in best_params.items()}
    pipeline = model_run.build_pipeline(
        n_estimators=stripped.get("n_estimators", config.N_ESTIMATORS),
        max_depth=stripped.get("max_depth", config.MAX_DEPTH),
        max_features=stripped.get("max_features", config.MAX_FEATURES),
    )
    pipeline = model_run.train_pipeline(pipeline, X_train, y_train)
    return pipeline


def step_evaluate_test_set(pipeline, X_test, y_test, test_samples):
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("STEP 4 – Evaluating on held-out test set")
    logger.info("=" * 60)

    y_pred, y_proba = model_run.predict(pipeline, X_test)
    metrics = evaluation.compute_metrics(y_test, y_pred, y_proba)
    evaluation.print_metrics_report(metrics, y_test, y_pred, label="Test Set")

    figs_dir = config.FIGURES_DIR
    evaluation.plot_confusion_matrix(
        y_test, y_pred, metrics,
        output_path=figs_dir / "confusion_matrix.png",
    )
    evaluation.plot_roc_curve(
        y_test, y_proba,
        output_path=figs_dir / "roc_curve.png",
    )
    evaluation.plot_feature_importance(
        pipeline, X_test.columns,
        top_n=15,
        output_path=figs_dir / "feature_importance_top15.png",
    )
    evaluation.plot_feature_importance_spectrum(
        pipeline, X_test.columns,
        output_path=figs_dir / "feature_importance_spectrum.png",
    )
    evaluation.plot_predicted_probabilities(
        y_test, y_pred, y_proba,
        output_path=figs_dir / "predicted_probabilities.png",
    )

    return metrics


def step_evaluate_validation(pipeline, train_columns):
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("STEP 5 – Evaluating on external validation data")
    logger.info("=" * 60)

    try:
        X_val, y_val, meta = data_prep.load_validation_data()
    except FileNotFoundError as exc:
        logger.warning("Validation data not found – skipping. (%s)", exc)
        return

    # Align validation features to the columns the pipeline was trained on.
    # Missing wavenumbers are filled with 0; extra wavenumbers are dropped.
    X_val = X_val.reindex(columns=train_columns, fill_value=0)
    logger.info("Validation features aligned to %d training columns", len(train_columns))

    metrics, results = evaluation.evaluate_validation_set(
        pipeline, X_val, y_val, meta,
        output_csv=config.ANALYSIS_DIR / "validation_results.csv",
    )
    evaluation.print_metrics_report(
        metrics, y_val, results["predicted"].values,
        label="External Validation Set",
    )

    # Per-sector accuracy breakdown (logged only)
    if "zoie_sector" in results.columns:
        logger.info("Sector-level accuracy:")
        sector_acc = (
            results.groupby("zoie_sector")["correct"]
            .agg(["sum", "count"])
            .assign(accuracy=lambda d: d["sum"] / d["count"])
            .rename(columns={"sum": "n_correct", "count": "n_total"})
        )
        logger.info("\n%s", sector_acc.to_string())


def step_sector_models(X_val, meta_val):
    """
    Train binary sector classifiers and evaluate on:
      - sectors_train      : sector model's own internal 30% hold-out (Cell 46)
      - sectors_validation : external validation data (Cell 57/58)
    """
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("STEP 6 – Binary sector classifiers")
    logger.info("=" * 60)

    try:
        all_with_sectors = data_prep.load_all_with_sectors()
    except Exception as exc:
        logger.warning("Sector model training skipped — could not load sector data: %s", exc)
        return

    # Matches notebook Cell 43: train_with_sectors.Sector.value_counts()
    counts = all_with_sectors["Sector"].value_counts()
    logger.info("Sector sample counts:\n%s", counts.to_string())
    print("\nSector sample counts (PLOPP training data):")
    print("-" * 35)
    for sector, n in counts.items():
        print(f"  {sector:<22} {n:>4}")
    print(f"  {'Total':<22} {counts.sum():>4}")
    print()

    # Train one binary model per sector (Cell 46 — all 7 sectors)
    sector_models = model_run.train_sector_binary_models(all_with_sectors)

    # Training hold-out: complex confusion matrix + accuracy bar chart (matches Cell 46)
    evaluation.print_sector_metrics_report(sector_models, label="Sector Models – Training Hold-out")
    evaluation.plot_binary_sector_confusion_matrices(
        sector_models,
        output_dir=config.FIGURES_DIR / "sectors_test",
        label_prefix="test_",
    )
    evaluation.plot_sector_accuracy_bar(
        sector_models,
        title="Total Accuracy Across Sectors",
        output_path=config.FIGURES_DIR / "sectors_test" / "accuracy_across_sectors.png",
    )

    # sectors_validation — external validation data (Cell 57/58)
    # The notebook evaluates all target sectors here; sectors without any
    # positives in the validation set still appear as all-negative rows.
    val_results = model_run.evaluate_sector_models_on_validation(
        sector_models, X_val, meta_val,
        sectors=config.VALIDATION_SECTORS,
    )
    evaluation.print_sector_metrics_report(val_results, label="Sector Models – External Validation")
    evaluation.plot_binary_sector_confusion_matrices(
        val_results,
        output_dir=config.FIGURES_DIR / "sectors_validation",
        label_prefix="validation_",
    )
    evaluation.plot_sector_accuracy_bar(
        val_results,
        title="Unseen Data Accuracy Comparison Across Sectors",
        output_path=config.FIGURES_DIR / "sectors_validation" / "accuracy_across_sectors.png",
    )


def step_save_pipeline(pipeline):
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("STEP 7 – Saving trained pipeline")
    logger.info("=" * 60)
    model_path = config.MODELS_DIR / "rf_pipeline.pkl"
    model_run.save_pipeline(pipeline, model_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level)
    logger = logging.getLogger("main")

    config.ensure_output_dirs()
    config.save_hyperparameters()
    logger.info("Output directories: figures=%s  models=%s", config.FIGURES_DIR, config.MODELS_DIR)

    X_train, y_train, X_test, y_test, test_samples = step_load_training_data()
    best_params = step_tune_pipeline(X_train, y_train, skip_tuning=args.skip_tuning)
    pipeline = step_train_pipeline(X_train, y_train, best_params)
    step_evaluate_test_set(pipeline, X_test, y_test, test_samples=test_samples)
    step_evaluate_validation(pipeline, X_train.columns)

    # Load validation data for sector model evaluation.
    # X_val is NOT reindexed to X_train.columns here — the sector models were
    # trained on PLOPP-only columns, so evaluate_sector_models_on_validation
    # handles alignment to each model's own feature_names_in_.
    try:
        X_val, y_val, meta_val = data_prep.load_validation_data()
        # Pass all validation samples — NaN-sector samples become class 0
        # ("not this sector") in each binary model, matching the notebook.
        step_sector_models(
            X_val=X_val.reset_index(drop=True),
            meta_val=meta_val.reset_index(drop=True),
        )
    except Exception as exc:
        logging.getLogger("main").warning("Sector step skipped: %s", exc)

    step_save_pipeline(pipeline)

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PLOPP/FLOPP spectral classification pipeline"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip GridSearchCV and load best params from models/hyperparameters.json",
    )
    main(parser.parse_args())
