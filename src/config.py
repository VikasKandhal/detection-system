"""
Configuration module for the Fraud Detection System.
Centralizes all paths, hyperparameters, feature lists, and thresholds.
"""

import os
from pathlib import Path

# ==============================================================================
# Project Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT  # Raw CSVs are in root
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for d in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Data Files
# ==============================================================================
TRAIN_TRANSACTION = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = RAW_DATA_DIR / "train_identity.csv"
TEST_TRANSACTION = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY = RAW_DATA_DIR / "test_identity.csv"

# ==============================================================================
# Random Seed (reproducibility)
# ==============================================================================
RANDOM_SEED = 42

# ==============================================================================
# Data Processing
# ==============================================================================
MISSING_THRESHOLD = 0.80       # Drop columns with >80% missing values
TEST_SIZE = 0.10               # Validation & test split ratio each
VALIDATION_SIZE = 0.10

# ==============================================================================
# Categorical Feature Lists (from the IEEE dataset)
# ==============================================================================
CATEGORICAL_FEATURES = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34",
    "id_35", "id_36", "id_37", "id_38",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"
]

# High-cardinality categorical features — use label encoding
HIGH_CARDINALITY_FEATURES = [
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo",
    "id_30", "id_31", "id_33"
]

# ==============================================================================
# Feature Engineering — Time Windows (in seconds, based on TransactionDT)
# ==============================================================================
TIME_WINDOWS = {
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "7d": 604800
}

# ==============================================================================
# Model Hyperparameter Search Spaces (Optuna defaults)
# ==============================================================================
OPTUNA_N_TRIALS = 30           # Number of Optuna optimization trials
OPTUNA_CV_FOLDS = 5            # Stratified K-Fold cross-validation folds

# XGBoost search space
XGBOOST_PARAMS = {
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 1000),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma": (0.0, 5.0),
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-8, 10.0),
}

# LightGBM search space
LIGHTGBM_PARAMS = {
    "num_leaves": (20, 150),
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 1000),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "lambda_l1": (1e-8, 10.0),
    "lambda_l2": (1e-8, 10.0),
    "min_child_samples": (5, 100),
}

# ==============================================================================
# Autoencoder Configuration
# ==============================================================================
AUTOENCODER_CONFIG = {
    "encoding_dim": 32,            # Bottleneck dimension
    "hidden_layers": [128, 64],    # Encoder hidden layer sizes
    "epochs": 50,
    "batch_size": 512,
    "learning_rate": 0.001,
    "validation_split": 0.15,
    "patience": 5,                 # Early stopping patience
}

# ==============================================================================
# Decision Threshold
# ==============================================================================
DEFAULT_THRESHOLD = 0.5           # Will be optimized during evaluation
PRECISION_TARGET = 0.90           # Target precision for threshold optimization
MIN_RECALL_TARGET = 0.50          # Minimum acceptable recall

# ==============================================================================
# API Configuration
# ==============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_NAME = "lightgbm_best"      # Default model to load for API serving
