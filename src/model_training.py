"""
Model Training for the Fraud Detection System.
Trains Logistic Regression (baseline), Random Forest, XGBoost, and LightGBM
with Optuna hyperparameter tuning and class imbalance handling.
"""

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, make_scorer
)
from typing import Dict, Tuple, Any

from src.config import (
    RANDOM_SEED, OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, MODELS_DIR,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS
)
from src.utils import timer, logger

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    exclude_cols: list = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features (X) and target (y)."""
    if exclude_cols is None:
        exclude_cols = ["isFraud", "TransactionID"]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    return X, y


@timer
def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series
) -> Dict[str, Any]:
    """
    Train Logistic Regression as a baseline model.
    
    Uses class_weight='balanced' to handle imbalance by automatically
    adjusting weights inversely proportional to class frequencies.
    """
    logger.info("Training Logistic Regression (baseline)...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="saga",          # Scalable for large datasets
        n_jobs=-1,
        C=1.0,
        penalty="l2"
    )

    model.fit(X_train, y_train)

    # Validation predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
    prauc = average_precision_score(y_val, y_pred_proba)
    roc = roc_auc_score(y_val, y_pred_proba)

    results = {
        "model": model,
        "name": "Logistic Regression",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": prauc,
        "roc_auc": roc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred
    }

    logger.info(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"  PR-AUC: {prauc:.4f} | ROC-AUC: {roc:.4f}")

    # Save model
    joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
    return results


@timer
def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series
) -> Dict[str, Any]:
    """
    Train Random Forest with balanced class weights.
    
    Uses balanced_subsample to resample each bootstrap with balanced weights,
    which is more effective than balanced class weights for ensemble methods.
    """
    logger.info("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt"
    )

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
    prauc = average_precision_score(y_val, y_pred_proba)
    roc = roc_auc_score(y_val, y_pred_proba)

    results = {
        "model": model,
        "name": "Random Forest",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": prauc,
        "roc_auc": roc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred
    }

    logger.info(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"  PR-AUC: {prauc:.4f} | ROC-AUC: {roc:.4f}")

    joblib.dump(model, MODELS_DIR / "random_forest.pkl")
    return results


@timer
def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    tune: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost with Optuna hyperparameter tuning.
    
    Uses scale_pos_weight to handle class imbalance:
    scale_pos_weight = count(negative) / count(positive)
    This tells XGBoost to pay more attention to the minority (fraud) class.
    """
    import xgboost as xgb

    logger.info("Training XGBoost...")

    # Calculate class imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logger.info(f"  Class ratio (neg/pos): {scale_pos_weight:.1f}")

    if tune:
        logger.info(f"  Running Optuna optimization ({OPTUNA_N_TRIALS} trials)...")

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", *XGBOOST_PARAMS["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *XGBOOST_PARAMS["learning_rate"], log=True),
                "n_estimators": trial.suggest_int("n_estimators", *XGBOOST_PARAMS["n_estimators"]),
                "subsample": trial.suggest_float("subsample", *XGBOOST_PARAMS["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *XGBOOST_PARAMS["colsample_bytree"]),
                "min_child_weight": trial.suggest_int("min_child_weight", *XGBOOST_PARAMS["min_child_weight"]),
                "gamma": trial.suggest_float("gamma", *XGBOOST_PARAMS["gamma"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *XGBOOST_PARAMS["reg_alpha"], log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", *XGBOOST_PARAMS["reg_lambda"], log=True),
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "aucpr",
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "tree_method": "hist",  # Fast histogram-based method
            }

            model = xgb.XGBClassifier(**params)
            
            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(
                model, X_train, y_train, cv=cv,
                scoring="average_precision", n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

        best_params = study.best_params
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["eval_metric"] = "aucpr"
        best_params["random_state"] = RANDOM_SEED
        best_params["n_jobs"] = -1
        best_params["tree_method"] = "hist"

        logger.info(f"  Best Optuna score: {study.best_value:.4f}")
        logger.info(f"  Best params: {best_params}")
    else:
        best_params = {
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    # Train final model with best params
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
    prauc = average_precision_score(y_val, y_pred_proba)
    roc = roc_auc_score(y_val, y_pred_proba)

    results = {
        "model": model,
        "name": "XGBoost",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": prauc,
        "roc_auc": roc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "best_params": best_params
    }

    logger.info(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"  PR-AUC: {prauc:.4f} | ROC-AUC: {roc:.4f}")

    joblib.dump(model, MODELS_DIR / "xgboost_best.pkl")
    return results


@timer
def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    tune: bool = True
) -> Dict[str, Any]:
    """
    Train LightGBM with Optuna hyperparameter tuning.
    
    LightGBM is often the best performer for tabular fraud detection:
    - Handles categorical features natively
    - Faster training than XGBoost
    - Better handling of large feature spaces
    
    Uses is_unbalance=True for automatic class weight adjustment.
    """
    import lightgbm as lgb

    logger.info("Training LightGBM...")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    if tune:
        logger.info(f"  Running Optuna optimization ({OPTUNA_N_TRIALS} trials)...")

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", *LIGHTGBM_PARAMS["num_leaves"]),
                "max_depth": trial.suggest_int("max_depth", *LIGHTGBM_PARAMS["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *LIGHTGBM_PARAMS["learning_rate"], log=True),
                "n_estimators": trial.suggest_int("n_estimators", *LIGHTGBM_PARAMS["n_estimators"]),
                "feature_fraction": trial.suggest_float("feature_fraction", *LIGHTGBM_PARAMS["feature_fraction"]),
                "bagging_fraction": trial.suggest_float("bagging_fraction", *LIGHTGBM_PARAMS["bagging_fraction"]),
                "lambda_l1": trial.suggest_float("lambda_l1", *LIGHTGBM_PARAMS["lambda_l1"], log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", *LIGHTGBM_PARAMS["lambda_l2"], log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", *LIGHTGBM_PARAMS["min_child_samples"]),
                "is_unbalance": True,
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "verbose": -1,
                "bagging_freq": 5,
                "metric": "average_precision",
            }

            model = lgb.LGBMClassifier(**params)
            
            cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(
                model, X_train, y_train, cv=cv,
                scoring="average_precision", n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

        best_params = study.best_params
        best_params["is_unbalance"] = True
        best_params["random_state"] = RANDOM_SEED
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1
        best_params["bagging_freq"] = 5

        logger.info(f"  Best Optuna score: {study.best_value:.4f}")
        logger.info(f"  Best params: {best_params}")
    else:
        best_params = {
            "num_leaves": 64,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "min_child_samples": 20,
            "is_unbalance": True,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
            "bagging_freq": 5,
        }

    # Train final model
    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)]  # Suppress output
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
    prauc = average_precision_score(y_val, y_pred_proba)
    roc = roc_auc_score(y_val, y_pred_proba)

    results = {
        "model": model,
        "name": "LightGBM",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": prauc,
        "roc_auc": roc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "best_params": best_params
    }

    logger.info(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"  PR-AUC: {prauc:.4f} | ROC-AUC: {roc:.4f}")

    joblib.dump(model, MODELS_DIR / "lightgbm_best.pkl")
    return results


@timer
def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tune: bool = True
) -> Dict[str, Dict]:
    """
    Train all models and return results.
    
    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        tune: Whether to run hyperparameter tuning.
    
    Returns:
        Dictionary of model results keyed by model name.
    """
    logger.info("=" * 70)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    X_train, y_train = get_feature_target_split(train_df)
    X_val, y_val = get_feature_target_split(val_df)

    logger.info(f"  Training features: {X_train.shape}")
    logger.info(f"  Training fraud rate: {y_train.mean()*100:.2f}%")
    logger.info(f"  Validation fraud rate: {y_val.mean()*100:.2f}%")

    # Handle any infinity or very large values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

    results = {}

    # 1. Logistic Regression (baseline)
    results["logistic_regression"] = train_logistic_regression(X_train, y_train, X_val, y_val)

    # 2. Random Forest
    results["random_forest"] = train_random_forest(X_train, y_train, X_val, y_val)

    # 3. XGBoost
    results["xgboost"] = train_xgboost(X_train, y_train, X_val, y_val, tune=tune)

    # 4. LightGBM
    results["lightgbm"] = train_lightgbm(X_train, y_train, X_val, y_val, tune=tune)

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10} {'ROC-AUC':>10}")
    logger.info("-" * 75)
    for name, res in results.items():
        logger.info(
            f"{res['name']:<25} {res['precision']:>10.4f} {res['recall']:>10.4f} "
            f"{res['f1']:>10.4f} {res['pr_auc']:>10.4f} {res['roc_auc']:>10.4f}"
        )
    logger.info("=" * 70)

    # Save results summary
    joblib.dump(results, MODELS_DIR / "all_model_results.pkl")
    return results
