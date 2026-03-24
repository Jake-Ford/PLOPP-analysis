"""
evaluation.py
=============
Metrics computation and publication-quality figure generation for the
PLOPP/FLOPP spectral classification pipeline.

Functions
---------
compute_metrics          –  accuracy, precision, recall, F1, confusion matrix
print_metrics_report     –  formatted console summary
plot_confusion_matrix    –  annotated heatmap with TPR/FPR/TNR/FNR rates
plot_roc_curve           –  ROC curve with AUC
plot_feature_importance  –  top-N feature importances (bar chart)
plot_feature_importance_spectrum  –  full-spectrum importance line plot
plot_predicted_probabilities  –  scatter + box plot of predicted probabilities
evaluate_validation_set  –  end-to-end validation evaluation with summary CSV

Usage
-----
    from evaluation import (
        compute_metrics, print_metrics_report,
        plot_confusion_matrix, plot_roc_curve,
        plot_feature_importance, plot_feature_importance_spectrum,
        plot_predicted_probabilities, evaluate_validation_set,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling defaults
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": config.FIGURE_DPI,
    }
)

_PALETTE = {config.LABEL_PLOPP: "#D95F02", config.LABEL_FLOPP: "#1B9E77"}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    pos_label: str = config.LABEL_PLOPP,
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true    : array-like  – ground-truth labels
    y_pred    : array-like  – predicted labels
    y_proba   : np.ndarray  – class probabilities, shape (n, 2)
    pos_label : str         – the positive class label

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1,
        tp, tn, fp, fn,
        tpr, tnr, fpr, fnr,
        roc_auc
    """
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, config.LABEL_FLOPP])
    tp, fn, fp, tn = cm.ravel()

    # ROC – find column index for positive class
    classes = sorted(set(y_true))
    pos_idx = classes.index(pos_label)

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba[:, pos_idx], pos_label=pos_label)
    roc_auc = auc(fpr_curve, tpr_curve)

    total = tp + tn + fp + fn
    return {
        "accuracy":  (tp + tn) / total,
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall":    recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1":        f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "tnr": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "roc_auc": roc_auc,
    }


def print_metrics_report(
    metrics: dict,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    label: str = "Evaluation",
) -> None:
    """Print a formatted metrics summary to stdout."""
    divider = "─" * 52
    print(f"\n{divider}")
    print(f"  {label}")
    print(divider)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1 Score   : {metrics['f1']:.4f}")
    print(f"  ROC AUC    : {metrics['roc_auc']:.4f}")
    print(divider)
    print(f"  True Positives  : {metrics['tp']}")
    print(f"  True Negatives  : {metrics['tn']}")
    print(f"  False Positives : {metrics['fp']}")
    print(f"  False Negatives : {metrics['fn']}")
    print(divider)
    print(f"  TPR (Sensitivity) : {metrics['tpr']:.4f}")
    print(f"  TNR (Specificity) : {metrics['tnr']:.4f}")
    print(f"  FPR               : {metrics['fpr']:.4f}")
    print(f"  FNR               : {metrics['fnr']:.4f}")
    print(divider)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    metrics: dict,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """
    Plot an annotated confusion matrix heatmap matching notebook Cell 20.

    Layout: rows/cols ordered [FLoPP, PLoPP].  Diagonal cells are white text;
    off-diagonal cells are black text.  Rate annotations (TNR/FPR/FNR/TPR) are
    overlaid at fixed positions with fontsize 24 bold.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels
    metrics     : dict returned by compute_metrics()
    output_path : optional file path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Order: FLoPP=0 (negative), PLoPP=1 (positive) — matches notebook Cell 20
    labels = [config.LABEL_FLOPP, config.LABEL_PLOPP]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP

    confusion_df = pd.DataFrame(
        cm,
        index=[f"Actual {config.LABEL_FLOPP}", f"Actual {config.LABEL_PLOPP}"],
        columns=[f"Predicted {config.LABEL_FLOPP}", f"Predicted {config.LABEL_PLOPP}"],
    )

    tnr = metrics["tnr"]
    fpr = metrics["fpr"]
    fnr = metrics["fnr"]
    tpr = metrics["tpr"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(confusion_df, annot=False, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=16, weight="bold")
    ax.set_ylabel("True Label", fontsize=16, weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, weight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, weight="bold")

    # n= count overlays — diagonal white, off-diagonal black
    for i in range(2):
        for j in range(2):
            color = "white" if (i == 0 and j == 0) or (i == 1 and j == 1) else "black"
            plt.text(j + 0.5, i + 0.5, f"n = {cm[i, j]}",
                     ha="center", va="center", color=color, fontsize=20, weight="bold")

    # Rate annotations at notebook-exact positions
    plt.text(0.5, 0.6, f"TNR = {tnr * 100:.2f}%",
             ha="center", va="center", fontsize=24, weight="bold", color="white")
    plt.text(1.5, 0.6, f"FPR = {fpr * 100:.2f}%",
             ha="center", va="center", fontsize=24, weight="bold")
    plt.text(0.5, 1.6, f"FNR = {fnr * 100:.2f}%",
             ha="center", va="center", fontsize=24, weight="bold")
    plt.text(1.5, 1.6, f"TPR = {tpr * 100:.2f}%",
             ha="center", va="center", fontsize=24, weight="bold", color="white")

    plt.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """
    Plot the ROC curve with AUC annotation.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_proba     : class probabilities, shape (n, 2); column 1 = P(PLoPP)
    output_path : optional save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.array(y_true)
    classes = sorted(set(y_true))
    pos_idx = classes.index(config.LABEL_PLOPP)

    fpr, tpr, _ = roc_curve(y_true, y_proba[:, pos_idx], pos_label=config.LABEL_PLOPP)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#D95F02", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_feature_importance(
    pipeline: Pipeline,
    feature_names: Sequence,
    top_n: int = 15,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of the top-N most important wavenumber features.

    Matches notebook Cell 21: grey bars, figsize=(12,6), fontsize=24 on all
    axis labels and tick labels, no title.

    Parameters
    ----------
    pipeline      : fitted Pipeline
    feature_names : original wavenumber feature names
    top_n         : number of features to display
    output_path   : optional save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    importances, approx_names = _get_spectral_importances(pipeline, feature_names)
    approx_importances_series = pd.Series(importances, index=approx_names)

    # Sort ascending then take last top_n — matches notebook Cell 21 exactly
    top_importances = approx_importances_series.sort_values(ascending=True)[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 6))
    top_importances.plot(kind="barh", ax=ax, color="grey")
    ax.set_ylabel("Feature", fontsize=24)
    ax.set_xlabel("Approximate Feature Importance", fontsize=24)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(), fontsize=24)
    ax.set_yticklabels(top_importances.index, fontsize=24)
    plt.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_feature_importance_spectrum(
    pipeline: Pipeline,
    feature_names: Sequence,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """
    Line plot of approximate feature importance across all wavenumbers.

    Matches notebook Cell 24: figsize=(14,6), grey line, x-axis inverted,
    no fill, fontsize=16 on labels and ticks, no title.

    Parameters
    ----------
    pipeline      : fitted Pipeline
    feature_names : original wavenumber feature names
    output_path   : optional save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    importances, approx_names = _get_spectral_importances(pipeline, feature_names)
    approx_importances_series = pd.Series(importances, index=approx_names)

    # Sort by index descending then invert axis — matches notebook Cell 24
    df = approx_importances_series.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df.values, color="grey", linestyle="-", linewidth=1.5)
    ax.invert_xaxis()
    ax.set_xlabel("Feature", fontsize=16)
    ax.set_ylabel("Approximate Feature Importance", fontsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    plt.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_predicted_probabilities(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """
    Scatter + box plot showing the distribution of P(PLoPP) by true class.

    Matches notebook Cell 25: scatterplot coloured by Predicted Class
    (blue=FLoPP, orange=PLoPP) overlaid on a coolwarm boxplot coloured by
    Actual Class.  figsize=(10,6), fontsize=24 on labels and ticks, grid on,
    legend in lower right.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels (used for point colouring)
    y_proba     : class probabilities, shape (n, 2)
    output_path : optional save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.array(y_true)
    classes = sorted(set(y_true))
    pos_idx = classes.index(config.LABEL_PLOPP)

    plot_df = pd.DataFrame(
        {
            "Predicted Probability": y_proba[:, pos_idx],
            "Predicted Class":       y_pred,
            "Actual Class":          y_true,
        }
    )

    class_order    = [config.LABEL_FLOPP, config.LABEL_PLOPP]
    scatter_palette = {config.LABEL_FLOPP: "blue", config.LABEL_PLOPP: "orange"}
    box_palette     = {config.LABEL_FLOPP: "#a8c4e0", config.LABEL_PLOPP: "#e8b49a"}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot (background) — one box per actual class
    sns.boxplot(
        data=plot_df,
        y="Predicted Probability",
        x="Actual Class",
        order=class_order,
        palette=box_palette,
        whis=1.5,
        fliersize=0,
        ax=ax,
    )

    # Jittered scatter (foreground) — coloured by predicted class
    sns.stripplot(
        data=plot_df,
        y="Predicted Probability",
        x="Actual Class",
        hue="Predicted Class",
        order=class_order,
        hue_order=class_order,
        palette=scatter_palette,
        size=5,
        alpha=0.7,
        jitter=False,
        ax=ax,
    )

    ax.set_ylabel("Predicted Probability of PLoPP", fontsize=18)
    ax.set_xlabel("Actual Class", fontsize=24)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax.grid(True)
    ax.legend(title="Predicted Class", loc="lower right", fontsize=24, title_fontsize=24)
    plt.tight_layout()
    _save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------

def evaluate_validation_set(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta: pd.DataFrame,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
    output_csv: Path | str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Run the fitted pipeline on external validation data, compute metrics,
    and (optionally) write per-sample results to a CSV.

    Parameters
    ----------
    pipeline    : fitted Pipeline
    X_val       : feature matrix
    y_val       : true labels
    meta        : DataFrame with Sample, zoie_sector, observed columns
    threshold   : classification threshold for PLoPP
    output_csv  : optional path to save the per-sample results CSV

    Returns
    -------
    metrics : dict  (same structure as compute_metrics())
    results : pd.DataFrame with columns Sample, zoie_sector, observed,
              predicted, P_PLoPP, correct
    """
    from model_run import predict  # local import to avoid circular dependency

    y_pred, y_proba = predict(pipeline, X_val, threshold=threshold)
    classes = pipeline.classes_
    pos_idx = list(classes).index(config.LABEL_PLOPP)

    metrics = compute_metrics(y_val, y_pred, y_proba, pos_label=config.LABEL_PLOPP)

    results = meta.copy().reset_index(drop=True)
    results["predicted"] = y_pred
    results["P_PLoPP"]   = y_proba[:, pos_idx].round(4)
    results["correct"]   = results["predicted"] == results["observed"]

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
        logger.info("Validation results saved → %s", output_csv)

    return metrics, results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_spectral_importances(
    pipeline: Pipeline,
    feature_names: Sequence,
) -> tuple[np.ndarray, list]:
    """
    Map Random Forest feature importances from PCA space back to the original
    wavenumber space via PCA component loadings (approximate reconstruction).

    Returns
    -------
    importances  : np.ndarray, shape (n_original_features,)
    approx_names : list of original feature names in corresponding order
    """
    pca        = pipeline.named_steps["pca"]
    classifier = pipeline.named_steps["classifier"]

    rf_importances  = classifier.feature_importances_   # shape (n_components,)
    pca_components  = np.abs(pca.components_)           # shape (n_components, n_features)

    # Weight each PCA component by its RF importance and sum across components
    spectral_importances = np.dot(rf_importances, pca_components)

    feature_names = list(feature_names)
    return spectral_importances, feature_names


def print_sector_metrics_report(sector_results: dict, label: str = "Sector Models") -> None:
    """
    Print a per-sector accuracy metrics summary to the terminal.

    Parameters
    ----------
    sector_results : dict returned by train_sector_binary_models()
                     or evaluate_sector_models_on_validation().
    label          : heading label for the report block
    """
    divider = "─" * 60
    print(f"\n{divider}")
    print(f"  {label}")
    print(divider)
    print(f"  {'Sector':<22} {'Acc':>6} {'TPR':>6} {'TNR':>6} {'FPR':>6} {'FNR':>6} {'n':>5}")
    print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")

    for sector, data in sector_results.items():
        y_true = np.array(data["y_test"])
        y_pred = np.array(data["y_pred"])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        acc = (tp + tn) / total if total > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"  {sector:<22} {acc:>6.1%} {tpr:>6.1%} {tnr:>6.1%} {fpr:>6.1%} {fnr:>6.1%} {total:>5}")

    print(divider)


def plot_binary_sector_confusion_matrices(
    sector_results: dict,
    output_dir: Path | str | None = None,
    label_prefix: str = "",
) -> None:
    """
    Generate one confusion matrix per sector from binary sector model results.

    Matches notebook Cell 49 / 57 exactly: figsize=(10,7), annot=False,
    cbar=False, n= counts (fontsize=20 bold, only TN cell white), rate
    annotations (fontsize=24 bold, only TNR white), title with sector name.
    "General Industrial" gets a line break in the title display.

    Parameters
    ----------
    sector_results : dict returned by model_run.train_sector_binary_models()
                     or model_run.evaluate_sector_models_on_validation().
                     Keys are sector names; values have 'y_test' and 'y_pred'.
    output_dir     : directory to save figures
    label_prefix   : prepended to each filename (e.g. 'train_' or 'validation_')
    """
    for sector, data in sector_results.items():
        y_true = np.array(data["y_test"])
        y_pred = np.array(data["y_pred"])

        labels = [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # "General Industrial" gets a line break in tick labels and title (matches Cell 49)
        display_sector     = sector.replace("General Industrial", "General\nIndustrial")
        not_display_sector = f"Not {sector}".replace("General Industrial", "General\nIndustrial")
        tick_labels = [not_display_sector, display_sector]

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            cm, annot=False, fmt="d", cmap="Blues", cbar=False,
            xticklabels=tick_labels, yticklabels=tick_labels, ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=16, weight="bold")
        ax.set_ylabel("True Label", fontsize=16, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, weight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, weight="bold")

        # n= counts — only (0,0) TN cell is white (matches Cell 49)
        for i in range(2):
            for j in range(2):
                color = "white" if (i == 0 and j == 0) else "black"
                plt.text(j + 0.5, i + 0.5, f"n = {cm[i, j]}",
                         ha="center", va="center", fontsize=20, weight="bold", color=color)

        # Rate annotations — TNR white, all others black (matches Cell 49)
        plt.text(0.5, 0.6, f"TNR = {tnr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold", color="white")
        plt.text(1.5, 0.6, f"FPR = {fpr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold")
        plt.text(0.5, 1.6, f"FNR = {fnr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold")
        plt.text(1.5, 1.6, f"TPR = {tpr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold")

        plt.title(f"Confusion Matrix for {display_sector} Sector")
        plt.tight_layout()

        if output_dir is not None:
            safe = sector.replace(" ", "_").replace("/", "-")
            _save_figure(fig, Path(output_dir) / f"{label_prefix}confusion_{safe}.png")

        plt.close(fig)

    logger.info("Binary sector confusion matrices saved to %s", output_dir)


def plot_sector_confusion_matrices(
    results: pd.DataFrame,
    sector_col: str = "zoie_sector",
    true_col: str = "observed",
    pred_col: str = "predicted",
    output_dir: Path | str | None = None,
    label_prefix: str = "",
) -> None:
    """
    Generate one confusion matrix per sector and save to *output_dir*.

    Parameters
    ----------
    results     : DataFrame with at minimum sector_col, true_col, pred_col
    sector_col  : column name holding the sector label
    true_col    : column name holding true class labels
    pred_col    : column name holding predicted class labels
    output_dir  : directory to save figures; skipped if None
    label_prefix: string prepended to filenames (e.g. 'test_' or 'validation_')
    """
    if sector_col not in results.columns:
        logger.warning("Sector column '%s' not found — skipping sector plots.", sector_col)
        return

    # Order: FLoPP=0 (negative), PLoPP=1 (positive) — matches main confusion matrix
    labels  = [config.LABEL_FLOPP, config.LABEL_PLOPP]
    sectors = sorted(results[sector_col].dropna().unique())

    for sector in sectors:
        subset = results[results[sector_col] == sector]
        if len(subset) == 0:
            continue

        y_true = subset[true_col].values
        y_pred = subset[pred_col].values

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if cm.shape != (2, 2):
            continue

        # cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        total = tn + fp + fn + tp
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        confusion_df = pd.DataFrame(
            cm,
            index=[f"Actual {config.LABEL_FLOPP}", f"Actual {config.LABEL_PLOPP}"],
            columns=[f"Predicted {config.LABEL_FLOPP}", f"Predicted {config.LABEL_PLOPP}"],
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(confusion_df, annot=False, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted Label", fontsize=16, weight="bold")
        ax.set_ylabel("True Label", fontsize=16, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, weight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, weight="bold")

        # n= counts — white text on dark cells, black on light cells
        vmax = cm.max()
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > vmax * 0.5 else "black"
                plt.text(j + 0.5, i + 0.5, f"n = {cm[i, j]}",
                         ha="center", va="center", color=color, fontsize=20, weight="bold")

        # Rate annotations — value-based coloring
        cell_color = lambda i, j: "white" if cm[i, j] > vmax * 0.5 else "black"
        plt.text(0.5, 0.6, f"TNR = {tnr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold", color=cell_color(0, 0))
        plt.text(1.5, 0.6, f"FPR = {fpr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold", color=cell_color(0, 1))
        plt.text(0.5, 1.6, f"FNR = {fnr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold", color=cell_color(1, 0))
        plt.text(1.5, 1.6, f"TPR = {tpr * 100:.2f}%",
                 ha="center", va="center", fontsize=24, weight="bold", color=cell_color(1, 1))

        plt.title(f"Confusion Matrix — {sector} Sector  (n={total})")
        plt.tight_layout()

        if output_dir is not None:
            safe_name = sector.replace(" ", "_").replace("/", "-")
            out_path = Path(output_dir) / f"{label_prefix}confusion_{safe_name}.png"
            _save_figure(fig, out_path)

        plt.close(fig)

    logger.info("Sector confusion matrices saved to %s", output_dir)


def _save_figure(fig: plt.Figure, output_path: Path | str | None) -> None:
    """Save *fig* to *output_path* if a path is provided."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=config.FIGURE_DPI,
            format=config.FIGURE_FORMAT,
            bbox_inches="tight",
        )
        logger.info("Figure saved → %s", output_path)
