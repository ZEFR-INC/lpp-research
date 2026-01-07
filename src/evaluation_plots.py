"""Visualization utilities for model evaluation metrics.

This module provides production-grade plotting functions for evaluating meta-model
performance. All plots follow consistent styling and are saved to disk for analysis
and reporting.

Key Functions:
    - plot_confusion_matrix: Visualize classification confusion matrix
    - plot_f1_scores: Compare F1 scores (binary, macro, per-class)
    - plot_roc_curve: ROC curve with AUC score
    - plot_all_metrics: Generate all evaluation plots in one call

Example:
    ```python
    from evaluation_plots import plot_all_metrics

    plot_all_metrics(
        y_true=y_test,
        y_pred=predictions,
        y_proba=probabilities,
        output_dir="results/plots",
        experiment_name="XGBoost_f1"
    )
    ```
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import structlog

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from typing import Dict, Optional

logger = structlog.getLogger(__name__)
METRIC_ROC_AUC = "ROC AUC"

# Set consistent style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    labels: Optional[list] = None,
) -> None:
    """Plot confusion matrix with optional row-wise normalization.

    If ``normalize`` is True the raw confusion matrix counts are converted to
    row-wise proportions. This is safer for class imbalance and mirrors sklearn's
    ``normalize='true'`` behaviour. Single-class inputs are handled gracefully by
    forcing a 2x2 matrix with zeros for the missing class so downstream code and
    visual layouts remain consistent.

    Args:
        y_true: Ground-truth labels (binary 0/1 expected but will coerce)
        y_pred: Predicted labels (binary 0/1 expected but will coerce)
        output_path: Where to save the PNG figure
        title: Plot title
        normalize: Whether to plot normalized (row-wise proportion) values
        labels: Optional explicit label ordering; defaults to [0, 1]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = [0, 1]

    # Ensure binary numeric arrays
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    # Handle single-class scenario by injecting the missing class label
    unique_true = set(np.unique(y_true_arr)) | set(np.unique(y_pred_arr))
    if len(unique_true) == 1:
        # Force labels to [0,1]
        labels = [0, 1]

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm, row_sums, where=row_sums != 0)
        fmt = ".2f"
        cbar_label = "Proportion"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        # Use explicit variable name for clarity (ruff E741)
        xticklabels=[str(label) for label in labels],
        yticklabels=[str(label) for label in labels],
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(
        "Confusion matrix saved",
        path=str(output_path),
        normalize=normalize,
    )


def plot_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "F1 Scores by Class",
) -> None:
    """Plot F1 scores (binary, macro, and per-class) as a bar chart.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        title: Plot title
    """
    # Calculate F1 scores
    f1_binary = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Prepare data
    scores = {
        "F1 (Binary)": f1_binary,
        "F1 (Macro)": f1_macro,
        "F1 (Incorrect)": f1_class_0,
        "F1 (Correct)": f1_class_1,
    }

    # Create bar plot
    _, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        list(range(len(scores))),
        [float(v) for v in scores.values()],
        color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
        alpha=0.8,
        edgecolor="black",
    )

    # Add value labels on bars
    for i, (bar, (name, value)) in enumerate(zip(bars, scores.items())):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(scores.keys(), rotation=0, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"F1 scores plot saved to {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    """Plot ROC curve with AUC score.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        output_path: Path to save the plot
        title: Plot title
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    # Create plot
    _, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        color="#2ecc71",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")

    # Styling
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curve saved to {output_path}")


def plot_core_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    output_path: Path,
    title: str = "Classification Metrics",
) -> Dict[str, float]:
    """Plot core classification metrics (F1, ROC AUC, Precision, Recall).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for class 1 (optional for ROC AUC)
        output_path: Output path
        title: Plot title

    Returns:
        Dict with metric names and values
    """
    metrics: Dict[str, float] = {}
    metrics["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["Accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["Balanced_Accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    if y_proba is not None:
        metrics[METRIC_ROC_AUC] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics[METRIC_ROC_AUC] = np.nan

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(metrics)),
        list(metrics.values()),
        color=["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#d92020", "#44ad65"],
        alpha=0.85,
        edgecolor="black",
    )
    for bar, value in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{value:.4f}" if not np.isnan(value) else "NA",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics.keys(), fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Core metrics plot saved to {output_path}")
    return metrics


def plot_label_flipping_comparison(
    gt_binary: np.ndarray,
    y_llm: np.ndarray,
    y_llm_after_flipping: np.ndarray,
    output_path: Path,
    title: str = "Label Flipping Impact",
) -> None:
    """Compare LLM original vs label-flipped predictions using meta-model guidance.

    Flipping rule: For samples where meta-model predicts the LLM is incorrect (label 0),
    invert the LLM's predicted label.

    Args:
        y_true: Ground-truth labels
        y_llm: Original LLM predictions
        meta_model_predictions: Meta-model predictions (1=LLM correct, 0=LLM incorrect)
        output_path: Path to save comparison plot
        title: Plot title

    Returns:
        Nested dict: {"original": {...}, "corrected": {...}}
    """
    metrics_original = {
        "F1": float(f1_score(gt_binary, y_llm, zero_division=0)),
        "F1 Macro": float(f1_score(gt_binary, y_llm, average="macro", zero_division=0)),
        "Precision": float(precision_score(gt_binary, y_llm, zero_division=0)),
        "Recall": float(recall_score(gt_binary, y_llm, zero_division=0)),
        "Accuracy": float(accuracy_score(gt_binary, y_llm)),
        "Balanced Accuracy": float(balanced_accuracy_score(gt_binary, y_llm)),
    }
    metrics_corrected = {
        "F1": float(f1_score(gt_binary, y_llm_after_flipping, zero_division=0)),
        "F1 Macro": float(
            f1_score(gt_binary, y_llm_after_flipping, average="macro", zero_division=0)
        ),
        "Precision": float(
            precision_score(gt_binary, y_llm_after_flipping, zero_division=0)
        ),
        "Recall": float(recall_score(gt_binary, y_llm_after_flipping, zero_division=0)),
        "Accuracy": float(accuracy_score(gt_binary, y_llm_after_flipping)),
        "Balanced Accuracy": float(
            balanced_accuracy_score(gt_binary, y_llm_after_flipping)
        ),
    }

    metrics_keys = list(metrics_original.keys())
    x = np.arange(len(metrics_keys))
    width = 0.38

    _, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2,
        [metrics_original[k] for k in metrics_keys],
        width,
        label="Original LLM",
        color="#95a5a6",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        [metrics_corrected[k] for k in metrics_keys],
        width,
        label="After Label Flipping",
        color="#27ae60",
        edgecolor="black",
    )

    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Annotate improvements
    for i, k in enumerate(metrics_keys):
        diff = metrics_corrected[k] - metrics_original[k]
        ax.text(
            x[i],
            max(metrics_original[k], metrics_corrected[k]) + 0.06,
            f"Δ {diff:+.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2c3e50",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys, fontsize=11)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Label flipping comparison plot saved to {output_path}")


def plot_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    output_dir: Path,
    experiment_name: str,
    gt_binary: Optional[np.ndarray] = None,
    y_llm_binary: Optional[np.ndarray] = None,
    y_llm_after_flipping_binary: Optional[np.ndarray] = None,
) -> None:
    """Generate standard evaluation plots and optional label flipping comparison.

    Core plots are always generated:
        - confusion_matrix.png
        - f1_scores.png
        - core_metrics.png

    Additional plots:
        - roc_curve.png (only if probabilities available; gracefully skipped on ValueError)
        - label_flipping_comparison.png (only if *_binary arrays provided & aligned)
        - Per-model prefixed versions of core plots (``{experiment_name}_confusion_matrix.png`` etc.)

    Args:
        y_true: Ground-truth meta-model target labels.
        y_pred: Meta-model binary predictions.
        y_proba: Meta-model probability estimates (optional for ROC plot).
        output_dir: Directory where plots will be written.
        experiment_name: Prefix for per-model plots.
        gt_binary: Ground truth task labels (optional for flipping comparison).
        y_llm_binary: Original LLM task predictions (optional for flipping comparison).
        y_llm_after_flipping_binary: LLM predictions after flipping (optional for comparison).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating evaluation plots",
        experiment=experiment_name,
        output_dir=str(output_dir),
    )

    # 1. Confusion Matrix (combined)
    # Per-model prefixed confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=output_dir / f"{experiment_name}_confusion_matrix.png",
        title=f"{experiment_name} Confusion Matrix",
    )

    # 2. F1 Scores (per-class + macro variants)
    plot_f1_scores(
        y_true=y_true,
        y_pred=y_pred,
        output_path=output_dir / f"{experiment_name}_f1_scores.png",
        title=f"{experiment_name} F1 Scores",
    )

    # 3. Core metrics (F1 / Precision / Recall / ROC AUC)
    plot_core_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        output_path=output_dir / f"{experiment_name}_core_metrics.png",
        title=f"{experiment_name} Classification Metrics",
    )

    # 4. ROC Curve (if probabilities available)
    if y_proba is not None:
        try:
            plot_roc_curve(
                y_true=y_true,
                y_proba=y_proba,
                output_path=output_dir / f"{experiment_name}_roc_curve.png",
                title=f"{experiment_name} ROC Curve",
            )
        except ValueError as e:
            logger.warning("Could not plot ROC curve", error=str(e))
    else:
        logger.info("y_proba not provided; skipping ROC curve plot")

    # 5. Label Flipping Comparison (if applicable)
    if (
        gt_binary is not None
        and y_llm_binary is not None
        and y_llm_after_flipping_binary is not None
        and len(gt_binary) == len(y_llm_binary) == len(y_llm_after_flipping_binary)
    ):
        plot_label_flipping_comparison(
            gt_binary=gt_binary,
            y_llm=y_llm_binary,
            y_llm_after_flipping=y_llm_after_flipping_binary,
            output_path=output_dir / "label_flipping_comparison.png",
            title="LLM Label Flipping Impact (Test Set)",
        )
    else:
        logger.info(
            "Skipping label flipping comparison",
            reason="Missing or length mismatch in *_binary arrays",
            gt_binary_provided=gt_binary is not None,
            y_llm_binary_provided=y_llm_binary is not None,
            y_llm_after_flipping_binary_provided=y_llm_after_flipping_binary
            is not None,
        )

    logger.info(
        "All evaluation plots saved",
        experiment=experiment_name,
        output_dir=str(output_dir),
    )


def plot_metrics_comparison(
    results_df,
    output_path: Path,
    metrics: Optional[list] = None,
    title: str = "Model Performance Comparison",
) -> None:
    """Plot comparison of multiple models across different metrics.

    Args:
        results_df: DataFrame with experiment results (from MetaModelPipeline)
        output_path: Path to save the plot
        metrics: List of metric column names to compare
        title: Plot title
    """
    if metrics is None:
        metrics = ["test_f1", "test_f1_macro", "test_roc_auc", "test_balanced_accuracy"]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]

    if not available_metrics:
        logger.warning("No valid metrics found for comparison plot")
        return

    # Create subplot for each metric
    fig, axes = plt.subplots(
        nrows=len(available_metrics),
        ncols=1,
        figsize=(12, 4 * len(available_metrics)),
    )

    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        # Sort by metric value
        sorted_df = results_df.sort_values(metric, ascending=False).head(10)

        # Create bar plot
        bars = ax.barh(
            range(len(sorted_df)),
            sorted_df[metric],
            color="#3498db",
            alpha=0.7,
            edgecolor="black",
        )

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_df[metric])):
            ax.text(
                value + 0.005,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.4f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["experiment_name"], fontsize=10)
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Model", fontsize=12)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Model comparison plot saved to {output_path}")
