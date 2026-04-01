"""
Main training pipeline for the Fraud Detection System.
Orchestrates: Data Loading -> EDA -> Preprocessing -> Feature Engineering
-> Model Training -> Autoencoder -> Evaluation -> Explainability

Usage:
    python scripts/run_pipeline.py                 # Full pipeline
    python scripts/run_pipeline.py --quick          # Quick mode (skip tuning)
    python scripts/run_pipeline.py --skip-eda       # Skip EDA plots
    python scripts/run_pipeline.py --skip-autoencoder  # Skip autoencoder
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to Python path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import joblib

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
from src.utils import logger, timer


def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection Training Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode: skip hyperparameter tuning")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA visualizations")
    parser.add_argument("--skip-autoencoder", action="store_true", help="Skip autoencoder training")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explainability")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for debugging")
    return parser.parse_args()


@timer
def main():
    args = parse_args()
    
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("FRAUD DETECTION SYSTEM - TRAINING PIPELINE")
    logger.info("=" * 70)
    if args.quick:
        logger.info("  Mode: QUICK (no hyperparameter tuning)")
    if args.nrows:
        logger.info(f"  Debug mode: loading {args.nrows} rows")

    # ================================================================
    # STEP 1: Load Data
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DATA LOADING")
    logger.info("=" * 70)
    
    from src.data_loader import load_and_merge_data, save_processed_data

    # Check if processed data already exists
    processed_path = PROCESSED_DATA_DIR / "train_merged.parquet"
    if processed_path.exists() and args.nrows is None:
        logger.info("Loading pre-processed data from parquet...")
        df = pd.read_parquet(processed_path)
        logger.info(f"Loaded {len(df)} rows from cache")
    else:
        df = load_and_merge_data(nrows=args.nrows)
        if args.nrows is None:
            save_processed_data(df, "train_merged")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

    # ================================================================
    # STEP 2: EDA
    # ================================================================
    if not args.skip_eda:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 70)
        
        from src.eda import run_full_eda
        run_full_eda(df)
    else:
        logger.info("\n[SKIPPED] EDA")

    # ================================================================
    # STEP 3: Preprocessing
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: PREPROCESSING")
    logger.info("=" * 70)
    
    from src.preprocessing import run_preprocessing_pipeline
    train_df, val_df, test_df, artifacts = run_preprocessing_pipeline(df)

    # Free memory from full dataset
    del df
    import gc
    gc.collect()

    # ================================================================
    # STEP 4: Feature Engineering
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    from src.feature_engineering import run_feature_engineering

    # Apply feature engineering to each split independently
    # (to prevent data leakage from future data)
    train_df = run_feature_engineering(train_df, fast_mode=True)
    val_df = run_feature_engineering(val_df, fast_mode=True)
    test_df = run_feature_engineering(test_df, fast_mode=True)

    # Ensure consistent columns across splits
    common_cols = list(set(train_df.columns) & set(val_df.columns) & set(test_df.columns))
    train_df = train_df[common_cols]
    val_df = val_df[common_cols]
    test_df = test_df[common_cols]

    logger.info(f"Final feature count: {len(common_cols) - 2} (excluding isFraud, TransactionID)")

    # Update artifacts with final feature list
    exclude_cols = ["isFraud", "TransactionID"]
    final_feature_cols = [c for c in common_cols if c not in exclude_cols]
    artifacts["feature_cols"] = final_feature_cols
    joblib.dump(artifacts, MODELS_DIR / "preprocessing_artifacts.pkl")

    # ================================================================
    # STEP 5: Model Training
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("=" * 70)
    
    from src.model_training import train_all_models
    
    model_results = train_all_models(
        train_df, val_df, 
        tune=(not args.quick)
    )

    # ================================================================
    # STEP 6: Autoencoder (Optional)
    # ================================================================
    if not args.skip_autoencoder:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: AUTOENCODER ANOMALY DETECTION")
        logger.info("=" * 70)
        
        try:
            from src.autoencoder import train_autoencoder
            from src.model_training import get_feature_target_split

            X_train, y_train = get_feature_target_split(train_df)
            X_val, y_val = get_feature_target_split(val_df)

            # Handle inf/nan
            X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

            ae_results = train_autoencoder(X_train, y_train, X_val, y_val)
            model_results["autoencoder"] = ae_results
        except ImportError as e:
            logger.warning(f"Skipping autoencoder (TensorFlow not available): {e}")
        except Exception as e:
            logger.warning(f"Autoencoder training failed: {e}")
    else:
        logger.info("\n[SKIPPED] Autoencoder")

    # ================================================================
    # STEP 7: Evaluation
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: EVALUATION")
    logger.info("=" * 70)
    
    from src.evaluation import run_full_evaluation
    from src.model_training import get_feature_target_split

    _, y_val = get_feature_target_split(val_df)
    
    optimized_results = run_full_evaluation(y_val.values, model_results)

    # Also evaluate on TEST set with the best model
    logger.info("\n--- Test Set Evaluation (Best Model) ---")
    from src.evaluation import evaluate_model, plot_confusion_matrix
    
    # Find the best model by PR-AUC
    best_name = max(
        optimized_results.items(),
        key=lambda x: x[1].get("pr_auc", 0)
    )[0]
    best_model = optimized_results[best_name]["model"]
    best_threshold = optimized_results[best_name].get("optimal_threshold", 0.5)
    
    X_test, y_test = get_feature_target_split(test_df)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get test predictions
    if best_name == "autoencoder":
        from src.autoencoder import get_anomaly_scores
        y_test_proba = get_anomaly_scores(X_test, best_model)
    else:
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_metrics = evaluate_model(y_test.values, y_test_proba, f"{optimized_results[best_name]['name']} (Test)", best_threshold)
    
    logger.info(f"\n  TEST SET RESULTS ({optimized_results[best_name]['name']}):")
    logger.info(f"    Precision: {test_metrics['precision']:.4f}")
    logger.info(f"    Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"    F1-Score:  {test_metrics['f1']:.4f}")
    logger.info(f"    PR-AUC:    {test_metrics['pr_auc']:.4f}")
    logger.info(f"    ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    plot_confusion_matrix(y_test.values, y_test_pred, f"{optimized_results[best_name]['name']} (Test)")

    # ================================================================
    # STEP 8: Explainability
    # ================================================================
    if not args.skip_shap:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 8: SHAP EXPLAINABILITY")
        logger.info("=" * 70)
        
        try:
            from src.explainability import run_explainability_pipeline

            # Use the best tree-based model for SHAP
            tree_models = {k: v for k, v in optimized_results.items() 
                         if k in ["xgboost", "lightgbm", "random_forest"]}
            
            if tree_models:
                best_tree_name = max(tree_models.items(), key=lambda x: x[1].get("pr_auc", 0))[0]
                best_tree_model = tree_models[best_tree_name]["model"]
                
                X_val_clean, y_val_clean = get_feature_target_split(val_df)
                X_val_clean = X_val_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                run_explainability_pipeline(
                    best_tree_model, X_val_clean, y_val_clean,
                    model_name=tree_models[best_tree_name]["name"]
                )
        except Exception as e:
            logger.warning(f"SHAP explainability failed: {e}")
    else:
        logger.info("\n[SKIPPED] SHAP Explainability")

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - pipeline_start
    minutes, seconds = divmod(total_time, 60)
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {int(minutes)}m {seconds:.1f}s")
    logger.info(f"  Models saved to: {MODELS_DIR}")
    logger.info(f"  Plots saved to: {FIGURES_DIR}")
    logger.info(f"  Best model: {optimized_results[best_name]['name']}")
    logger.info(f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"  Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Test Recall: {test_metrics['recall']:.4f}")
    logger.info("=" * 70)
    logger.info("\nTo start the API server:")
    logger.info("  python scripts/run_api.py")
    logger.info("\nTo start the React frontend:")
    logger.info("  cd frontend && npm run dev")


if __name__ == "__main__":
    main()
