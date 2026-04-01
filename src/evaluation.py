"""
Evaluation module for the Fraud Detection System.
Computes metrics, plots confusion matrices, PR curves,
and optimizes decision thresholds for high precision.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from typing import Dict, List, Tuple
import json

from src.config import FIGURES_DIR, PRECISION_TARGET, MIN_RECALL_TARGET, MODELS_DIR
from src.utils import timer, logger


plt.style.use("seaborn-v0_8-whitegrid")


@timer
def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    threshold: float = 0.5
) -> Dict:
    """
    Comprehensive evaluation of a single model.
    
    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        model_name: Name for logging/saving.
        threshold: Decision threshold.
    
    Returns:
        Dictionary of all metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    prauc = average_precision_score(y_true, y_pred_proba)
    roc = roc_auc_score(y_true, y_pred_proba)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(prauc),
        "roc_auc": float(roc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_fraud": int(y_true.sum()),
        "total_legit": int((1 - y_true).sum()),
        "fraud_caught": int(tp),
        "fraud_missed": int(fn),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
    }
    
    return metrics


@timer
def optimize_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    precision_target: float = None,
    min_recall: float = None
) -> Tuple[float, Dict]:
    """
    Optimize decision threshold to maximize precision while maintaining recall.
    
    Strategy:
    1. Find all thresholds that achieve >= precision_target
    2. Among those, select the one with highest recall
    3. If no threshold meets the precision target, find the best F1 threshold
    
    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        model_name: Model name for logging.
        precision_target: Target precision (default from config).
        min_recall: Minimum acceptable recall (default from config).
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold).
    """
    if precision_target is None:
        precision_target = PRECISION_TARGET
    if min_recall is None:
        min_recall = MIN_RECALL_TARGET

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Strategy 1: Find threshold where precision >= target AND recall >= min_recall
    valid_mask = (precisions[:-1] >= precision_target) & (recalls[:-1] >= min_recall)
    
    if valid_mask.any():
        # Among valid thresholds, maximize recall (we want to catch more fraud)
        valid_recalls = recalls[:-1][valid_mask]
        valid_thresholds = thresholds[valid_mask]
        best_idx = np.argmax(valid_recalls)
        optimal_threshold = valid_thresholds[best_idx]
        logger.info(
            f"  {model_name}: Found threshold meeting targets "
            f"(precision>={precision_target}, recall>={min_recall})"
        )
    else:
        # Strategy 2: Find threshold that maximizes F1-score
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        logger.info(
            f"  {model_name}: Could not meet both targets. Using best F1 threshold."
        )

    # Evaluate at optimal threshold
    metrics = evaluate_model(y_true, y_pred_proba, model_name, optimal_threshold)
    
    logger.info(
        f"  Optimal threshold: {optimal_threshold:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )

    return optimal_threshold, metrics


@timer
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir=None
):
    """Plot a detailed confusion matrix heatmap."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize for display
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations with both count and percentage
    annotations = np.array([
        [f"{cm[i][j]:,}\n({cm_normalized[i][j]*100:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    sns.heatmap(
        cm, annot=annotations, fmt="", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax, cbar_kws={"label": "Count"},
        linewidths=2, linecolor="white",
        annot_kws={"size": 14}
    )
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=15, fontweight="bold")

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(save_dir / f"cm_{safe_name}.png")
    plt.close()
    logger.info(f"  Saved: cm_{safe_name}.png")


@timer
def plot_precision_recall_curve(
    y_true: np.ndarray,
    model_results: Dict[str, Dict],
    save_dir=None
):
    """Plot Precision-Recall curves for all models on the same chart."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (name, results) in enumerate(model_results.items()):
        y_proba = results.get("y_pred_proba", results.get("reconstruction_error"))
        if y_proba is None:
            continue
            
        precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
        prauc = average_precision_score(y_true, y_proba)
        
        ax.plot(
            recalls, precisions,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{results['name']} (PR-AUC={prauc:.4f})"
        )

    # Add baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.4f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curves - Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "pr_curves_comparison.png")
    plt.close()
    logger.info("  Saved: pr_curves_comparison.png")


@timer
def plot_roc_curves(
    y_true: np.ndarray,
    model_results: Dict[str, Dict],
    save_dir=None
):
    """Plot ROC curves for all models."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (name, results) in enumerate(model_results.items()):
        y_proba = results.get("y_pred_proba", results.get("reconstruction_error"))
        if y_proba is None:
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_score = roc_auc_score(y_true, y_proba)
        
        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{results['name']} (AUC={roc_score:.4f})"
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves - Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves_comparison.png")
    plt.close()
    logger.info("  Saved: roc_curves_comparison.png")


@timer 
def plot_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    save_dir=None
):
    """Plot how precision and recall change with threshold."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    thresholds_range = np.arange(0.1, 0.95, 0.01)
    precisions_list = []
    recalls_list = []
    f1s_list = []

    for t in thresholds_range:
        y_pred = (y_pred_proba >= t).astype(int)
        if y_pred.sum() == 0:
            precisions_list.append(0)
            recalls_list.append(0)
            f1s_list.append(0)
            continue
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        precisions_list.append(p)
        recalls_list.append(r)
        f1s_list.append(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds_range, precisions_list, "b-", linewidth=2, label="Precision")
    ax.plot(thresholds_range, recalls_list, "r-", linewidth=2, label="Recall")
    ax.plot(thresholds_range, f1s_list, "g--", linewidth=2, label="F1-Score")
    
    ax.axhline(y=PRECISION_TARGET, color="blue", linestyle=":", alpha=0.4, label=f"Precision target ({PRECISION_TARGET})")
    ax.axhline(y=MIN_RECALL_TARGET, color="red", linestyle=":", alpha=0.4, label=f"Min recall ({MIN_RECALL_TARGET})")

    ax.set_xlabel("Decision Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Threshold Analysis - {model_name}", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(save_dir / f"threshold_analysis_{safe_name}.png")
    plt.close()
    logger.info(f"  Saved: threshold_analysis_{safe_name}.png")


@timer
def generate_model_comparison_table(
    model_results: Dict[str, Dict],
    save_dir=None
) -> pd.DataFrame:
    """Generate and save a comparison table of all models."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    rows = []
    for name, results in model_results.items():
        rows.append({
            "Model": results["name"],
            "Precision": results["precision"],
            "Recall": results["recall"],
            "F1-Score": results["f1"],
            "PR-AUC": results["pr_auc"],
            "ROC-AUC": results["roc_auc"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    
    # Save as CSV
    df.to_csv(save_dir / "model_comparison.csv", index=False)
    
    # Save as JSON for the API
    df.to_json(MODELS_DIR / "model_comparison.json", orient="records", indent=2)
    
    logger.info("\nModel Comparison Table:")
    logger.info(df.to_string(index=False))
    
    return df


@timer
def run_full_evaluation(
    y_true: np.ndarray,
    model_results: Dict[str, Dict],
    save_dir=None
) -> Dict[str, Dict]:
    """
    Run complete evaluation pipeline for all models.
    
    Args:
        y_true: True validation/test labels.
        model_results: Dictionary of model results.
        save_dir: Directory to save figures.
    
    Returns:
        Updated model results with optimal thresholds and metrics.
    """
    logger.info("=" * 70)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 70)

    if save_dir is None:
        save_dir = FIGURES_DIR

    optimized_results = {}
    thresholds = {}

    for name, results in model_results.items():
        y_proba = results.get("y_pred_proba", results.get("reconstruction_error"))
        if y_proba is None:
            continue
            
        # Optimize threshold
        opt_threshold, opt_metrics = optimize_threshold(
            y_true, y_proba, results["name"]
        )
        
        # Update results
        results["optimal_threshold"] = opt_threshold
        results["optimized_metrics"] = opt_metrics
        optimized_results[name] = results
        thresholds[name] = opt_threshold

        # Plot confusion matrix at optimal threshold
        y_pred_opt = (y_proba >= opt_threshold).astype(int)
        plot_confusion_matrix(y_true, y_pred_opt, results["name"], save_dir)

        # Plot threshold analysis
        plot_threshold_analysis(y_true, y_proba, results["name"], save_dir)

    # Plot PR curves and ROC curves
    plot_precision_recall_curve(y_true, optimized_results, save_dir)
    plot_roc_curves(y_true, optimized_results, save_dir)

    # Generate comparison table
    generate_model_comparison_table(optimized_results, save_dir)

    # Save optimal thresholds
    import joblib
    joblib.dump(thresholds, MODELS_DIR / "optimal_thresholds.pkl")

    # Find and report best model
    best_model = max(optimized_results.items(), key=lambda x: x[1]["pr_auc"])
    logger.info(f"\n  Best model by PR-AUC: {best_model[1]['name']} ({best_model[1]['pr_auc']:.4f})")

    return optimized_results
