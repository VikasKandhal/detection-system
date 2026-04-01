"""
Data Preprocessing for the IEEE Fraud Detection Dataset.
Handles missing values, categorical encoding, feature scaling, and
time-based train/validation/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import joblib

from src.config import (
    MISSING_THRESHOLD, CATEGORICAL_FEATURES, HIGH_CARDINALITY_FEATURES,
    RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE, MODELS_DIR, PROCESSED_DATA_DIR
)
from src.utils import timer, logger


@timer
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with a multi-strategy approach:
    1. Drop columns with >80% missing (too little signal).
    2. Fill numeric columns with median (robust to outliers).
    3. Fill categorical columns with 'Unknown'.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with missing values handled.
    """
    initial_cols = len(df.columns)
    
    # Step 1: Drop columns exceeding missing threshold
    missing_pct = df.isnull().sum() / len(df)
    drop_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    
    # Never drop target or key identifier columns
    protected = ["isFraud", "TransactionID", "TransactionDT", "TransactionAmt"]
    drop_cols = [c for c in drop_cols if c not in protected]
    
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"  Dropped {len(drop_cols)} columns with >{MISSING_THRESHOLD*100:.0f}% missing")

    # Step 2: Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Step 3: Fill categorical columns with 'Unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    final_cols = len(df.columns)
    logger.info(f"  Columns: {initial_cols} -> {final_cols} ({initial_cols - final_cols} dropped)")
    logger.info(f"  Remaining missing values: {df.isnull().sum().sum()}")

    return df


@timer
def encode_categorical_features(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using Label Encoding.
    
    For high-cardinality features (like DeviceInfo with 1000+ unique values),
    we use label encoding rather than one-hot to avoid feature explosion.
    For tree-based models (XGBoost, LightGBM), label encoding works well.
    
    Args:
        df: Input DataFrame.
        encoders: Pre-fitted encoders (for inference). None = fit new ones.
        fit: Whether to fit the encoders (True for training, False for inference).
    
    Returns:
        Tuple of (encoded DataFrame, fitted encoders dict).
    """
    if encoders is None:
        encoders = {}

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"  Encoding {len(cat_cols)} categorical columns")

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            # Handle unseen categories by adding 'Unknown' to classes
            df[col] = df[col].astype(str)
            le.fit(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                logger.warning(f"  No encoder found for column: {col}, using label encoding")
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                le.fit(df[col])
                encoders[col] = le

        # Transform, handling unseen labels
        df[col] = df[col].astype(str)
        known_classes = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known_classes else "Unknown")
        
        # Re-fit if "Unknown" wasn't in original classes
        if "Unknown" not in known_classes:
            le.classes_ = np.append(le.classes_, "Unknown")
        
        df[col] = le.transform(df[col])

    return df, encoders


@timer
def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to numeric features.
    Required for Logistic Regression and Autoencoder, optional for tree models.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of columns to scale.
        scaler: Pre-fitted scaler (for inference). None = fit new one.
        fit: Whether to fit the scaler.
    
    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


@timer
def create_time_based_split(
    df: pd.DataFrame,
    target_col: str = "isFraud"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/validation/test splits.
    
    This is CRITICAL for fraud detection to prevent data leakage:
    - Train on earlier transactions
    - Validate/test on later transactions
    This simulates real-world deployment where the model only sees past data.
    
    Args:
        df: Input DataFrame (must have TransactionDT).
        target_col: Target column name.
    
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Sort by TransactionDT for time-based split
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * (1 - TEST_SIZE - VALIDATION_SIZE))
    val_end = int(n * (1 - TEST_SIZE))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"  Time-based split:")
    logger.info(f"    Train: {len(train_df):,} rows ({train_df[target_col].mean()*100:.2f}% fraud)")
    logger.info(f"    Val:   {len(val_df):,} rows ({val_df[target_col].mean()*100:.2f}% fraud)")
    logger.info(f"    Test:  {len(test_df):,} rows ({test_df[target_col].mean()*100:.2f}% fraud)")

    # Log time ranges
    logger.info(f"    Train DT range: [{train_df['TransactionDT'].min()}, {train_df['TransactionDT'].max()}]")
    logger.info(f"    Val DT range:   [{val_df['TransactionDT'].min()}, {val_df['TransactionDT'].max()}]")
    logger.info(f"    Test DT range:  [{test_df['TransactionDT'].min()}, {test_df['TransactionDT'].max()}]")

    return train_df, val_df, test_df


@timer
def run_preprocessing_pipeline(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the complete preprocessing pipeline:
    1. Handle missing values
    2. Encode categorical features
    3. Time-based split
    4. Scale features (returns scaler for inference)
    
    Args:
        df: Raw merged DataFrame.
    
    Returns:
        Tuple of (train_df, val_df, test_df, artifacts_dict).
        artifacts_dict contains fitted encoders, scaler, and feature lists.
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    # Step 1: Handle missing values
    df = handle_missing_values(df)

    # Step 2: Encode categorical features
    df, encoders = encode_categorical_features(df)

    # Step 3: Time-based split
    train_df, val_df, test_df = create_time_based_split(df)

    # Step 4: Identify feature columns (everything except target and ID)
    exclude_cols = ["isFraud", "TransactionID"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    numeric_feature_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Step 5: Scale features (fit on train only to prevent leakage)
    scaler = StandardScaler()
    train_scaled = train_df[numeric_feature_cols].copy()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_scaled),
        columns=numeric_feature_cols,
        index=train_df.index
    )
    
    # Save artifacts for inference
    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "numeric_feature_cols": numeric_feature_cols,
    }

    # Save preprocessing artifacts
    joblib.dump(artifacts, MODELS_DIR / "preprocessing_artifacts.pkl")
    logger.info(f"  Saved preprocessing artifacts to {MODELS_DIR / 'preprocessing_artifacts.pkl'}")

    return train_df, val_df, test_df, artifacts
