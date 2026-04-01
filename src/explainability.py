"""
Explainability module using SHAP (SHapley Additive exPlanations).
Provides global feature importance and local per-transaction explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from typing import Dict, List, Any

from src.config import FIGURES_DIR, MODELS_DIR
from src.utils import timer, logger


@timer
def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str = "LightGBM",
    max_samples: int = 5000
) -> shap.Explanation:
    """
    Compute SHAP values for a trained model.
    
    Uses TreeExplainer for tree-based models (XGBoost/LightGBM/RF)
    which is exact and efficient. Falls back to KernelExplainer for
    linear models.
    
    Args:
        model: Trained model.
        X: Feature matrix.
        model_name: Name for logging.
        max_samples: Max samples for SHAP computation (memory limit).
    
    Returns:
        SHAP Explanation object.
    """
    logger.info(f"Computing SHAP values for {model_name}...")

    # Subsample for efficiency if dataset is large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
        logger.info(f"  Subsampled to {max_samples} rows for SHAP computation")
    else:
        X_sample = X

    # Handle inf/nan
    X_sample = X_sample.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Choose the right explainer based on model type
    model_type = type(model).__name__
    
    if model_type in ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
    else:
        # For logistic regression and other models
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer(X_sample)

    logger.info(f"  SHAP values computed: shape {shap_values.shape}")
    return shap_values


@timer
def plot_global_feature_importance(
    shap_values: shap.Explanation,
    model_name: str,
    max_features: int = 25,
    save_dir=None
):
    """
    Plot global feature importance using SHAP summary plot.
    
    Shows which features have the largest impact on predictions
    across all transactions — crucial for model interpretability
    and regulatory compliance in fraud detection.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR

    # Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values, max_display=max_features,
        show=False, plot_size=(12, 10)
    )
    plt.title(f"SHAP Feature Importance - {model_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(save_dir / f"shap_summary_{safe_name}.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: shap_summary_{safe_name}.png")

    # Bar plot (mean absolute SHAP value)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values, max_display=max_features,
        plot_type="bar", show=False, plot_size=(12, 10)
    )
    plt.title(f"Mean |SHAP| Feature Importance - {model_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_bar_{safe_name}.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: shap_bar_{safe_name}.png")


@timer
def plot_single_prediction_explanation(
    shap_values: shap.Explanation,
    idx: int,
    model_name: str,
    save_dir=None
):
    """
    Plot SHAP waterfall for a single transaction prediction.
    Shows exactly why a specific transaction was flagged as fraud.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.waterfall_plot(shap_values[idx], max_display=15, show=False)
    plt.title(f"Transaction #{idx} - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(save_dir / f"shap_waterfall_{safe_name}_{idx}.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: shap_waterfall_{safe_name}_{idx}.png")


def explain_prediction(
    model,
    X_single: pd.DataFrame,
    feature_names: List[str] = None,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Generate a human-readable explanation for a single transaction prediction.
    
    This is used by the API to explain why a transaction was flagged.
    
    Args:
        model: Trained model.
        X_single: Single transaction features (1 row DataFrame).
        feature_names: List of feature names.
        top_n: Number of top contributing features to show.
    
    Returns:
        Dictionary with explanation details.
    """
    # Handle inf/nan
    X_clean = X_single.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get prediction
    pred_proba = model.predict_proba(X_clean)[0, 1]

    # Compute SHAP values for this single prediction
    model_type = type(model).__name__
    if model_type in ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_clean)
    
    shap_values = explainer.shap_values(X_clean)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Binary classification: take values for positive class
        sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        sv = shap_values[0]

    if feature_names is None:
        feature_names = list(X_single.columns)

    # Get top contributing features
    feature_importance = list(zip(feature_names, sv, X_clean.values[0]))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = []
    for feat_name, shap_val, feat_val in feature_importance[:top_n]:
        direction = "increases" if shap_val > 0 else "decreases"
        top_features.append({
            "feature": feat_name,
            "value": float(feat_val),
            "shap_value": float(shap_val),
            "direction": direction,
            "impact": f"{feat_name}={feat_val:.4f} {direction} fraud risk by {abs(shap_val):.4f}"
        })

    # Generate human-readable explanation
    reasons = []
    for f in top_features[:5]:
        if f["shap_value"] > 0:
            reasons.append(f"⚠️ {f['feature']} = {f['value']:.4f} (increases fraud risk)")
        else:
            reasons.append(f"✅ {f['feature']} = {f['value']:.4f} (decreases fraud risk)")

    explanation = {
        "fraud_probability": float(pred_proba),
        "top_risk_factors": top_features,
        "human_readable_explanation": reasons,
        "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
    }

    return explanation


@timer
def run_explainability_pipeline(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "LightGBM",
    save_dir=None
) -> shap.Explanation:
    """
    Run the full explainability pipeline:
    1. Compute SHAP values
    2. Plot global feature importance
    3. Plot individual fraud case explanations
    
    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation labels.
        model_name: Model name for labeling.
    
    Returns:
        SHAP values object.
    """
    logger.info("=" * 70)
    logger.info("EXPLAINABILITY PIPELINE")
    logger.info("=" * 70)

    if save_dir is None:
        save_dir = FIGURES_DIR

    # Compute SHAP values
    shap_values = compute_shap_values(model, X_val, model_name)

    # Global importance plots
    plot_global_feature_importance(shap_values, model_name, save_dir=save_dir)

    # Plot explanations for specific fraud cases
    fraud_indices = np.where(y_val.values == 1)[0]
    if len(fraud_indices) > 0:
        # Explain first 3 fraud cases
        for i, fraud_idx in enumerate(fraud_indices[:3]):
            # Map the fraud_idx from the original index to the subsampled index if needed
            if fraud_idx < len(shap_values):
                plot_single_prediction_explanation(
                    shap_values, fraud_idx, model_name, save_dir
                )

    # Save SHAP values
    joblib.dump(shap_values, MODELS_DIR / f"shap_values_{model_name.lower().replace(' ', '_')}.pkl")

    return shap_values
