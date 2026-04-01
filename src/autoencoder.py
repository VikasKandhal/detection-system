"""
Autoencoder-based Anomaly Detection for Fraud Detection.
Trains on legitimate transactions only — fraud shows up as high reconstruction error.
CPU-optimized implementation.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score
)
from typing import Tuple, Dict, Any

from src.config import AUTOENCODER_CONFIG, RANDOM_SEED, MODELS_DIR
from src.utils import timer, logger


@timer
def build_autoencoder(input_dim: int):
    """
    Build an autoencoder model for anomaly detection.
    
    Architecture:
        Encoder: input -> 128 -> 64 -> 32 (bottleneck)
        Decoder: 32 -> 64 -> 128 -> input
    
    The bottleneck forces the model to learn a compressed representation
    of normal transactions. Fraud transactions, being different from normal
    patterns, will have high reconstruction error.
    
    Args:
        input_dim: Number of input features.
    
    Returns:
        Compiled autoencoder model.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    config = AUTOENCODER_CONFIG
    hidden_layers = config["hidden_layers"]
    encoding_dim = config["encoding_dim"]

    # Set random seed for reproducibility
    tf.random.set_seed(RANDOM_SEED)

    # Encoder
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(
            units, activation="relu",
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"encoder_{i}"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

    # Bottleneck
    encoded = layers.Dense(
        encoding_dim, activation="relu",
        name="bottleneck"
    )(x)

    # Decoder (mirror of encoder)
    x = encoded
    for i, units in enumerate(reversed(hidden_layers)):
        x = layers.Dense(
            units, activation="relu",
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"decoder_{i}"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

    # Output layer (reconstructing input)
    decoded = layers.Dense(input_dim, activation="linear", name="output")(x)

    autoencoder = keras.Model(inputs, decoded, name="fraud_autoencoder")
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="mse"  # Mean Squared Error for reconstruction
    )

    logger.info(f"  Autoencoder built: {input_dim} -> {hidden_layers} -> {encoding_dim} -> {list(reversed(hidden_layers))} -> {input_dim}")
    logger.info(f"  Total parameters: {autoencoder.count_params():,}")

    return autoencoder


@timer
def train_autoencoder(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Any]:
    """
    Train autoencoder on LEGITIMATE transactions only (semi-supervised).
    
    Key insight: By training only on normal transactions, the autoencoder
    learns to reconstruct normal patterns well. Fraudulent transactions
    will have high reconstruction error, making them detectable.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
    
    Returns:
        Dictionary with model, scaler, threshold, and metrics.
    """
    import tensorflow as tf
    from tensorflow import keras

    config = AUTOENCODER_CONFIG
    
    logger.info("Training Autoencoder for Anomaly Detection...")

    # Step 1: Filter to legitimate transactions only for training
    X_train_legit = X_train[y_train == 0].copy()
    logger.info(f"  Training on {len(X_train_legit)} legitimate transactions")

    # Step 2: Scale features (autoencoder requires normalized input)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_legit)
    X_val_scaled = scaler.transform(X_val)

    # Replace any NaN/Inf after scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)

    input_dim = X_train_scaled.shape[1]

    # Step 3: Build and train autoencoder
    autoencoder = build_autoencoder(input_dim)

    # Early stopping to prevent overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True
        )
    ]

    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,  # Input = Output (reconstruction)
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=config["validation_split"],
        callbacks=callbacks,
        verbose=1
    )

    # Step 4: Calculate reconstruction error on validation set
    X_val_pred = autoencoder.predict(X_val_scaled, batch_size=config["batch_size"])
    reconstruction_error = np.mean(np.power(X_val_scaled - X_val_pred, 2), axis=1)

    # Step 5: Find optimal threshold using validation labels
    # Use the threshold that maximizes F1-score
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, reconstruction_error)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]

    # Step 6: Evaluate with optimal threshold
    y_pred = (reconstruction_error > optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
    prauc = average_precision_score(y_val, reconstruction_error)
    roc = roc_auc_score(y_val, reconstruction_error)

    results = {
        "model": autoencoder,
        "scaler": scaler,
        "threshold": optimal_threshold,
        "name": "Autoencoder",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": prauc,
        "roc_auc": roc,
        "reconstruction_error": reconstruction_error,
        "y_pred": y_pred,
        "history": history.history,
    }

    logger.info(f"  Optimal threshold: {optimal_threshold:.6f}")
    logger.info(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"  PR-AUC: {prauc:.4f} | ROC-AUC: {roc:.4f}")

    # Save artifacts
    autoencoder.save(str(MODELS_DIR / "autoencoder.keras"))
    joblib.dump(scaler, MODELS_DIR / "autoencoder_scaler.pkl")
    joblib.dump(optimal_threshold, MODELS_DIR / "autoencoder_threshold.pkl")
    logger.info(f"  Saved autoencoder model and artifacts")

    return results


def get_anomaly_scores(
    X: pd.DataFrame,
    autoencoder=None,
    scaler=None
) -> np.ndarray:
    """
    Get anomaly scores (reconstruction error) for new transactions.
    
    Args:
        X: Features DataFrame.
        autoencoder: Trained autoencoder model.
        scaler: Fitted StandardScaler.
    
    Returns:
        Array of reconstruction errors (anomaly scores).
    """
    if autoencoder is None:
        from tensorflow import keras
        autoencoder = keras.models.load_model(str(MODELS_DIR / "autoencoder.keras"))
    if scaler is None:
        scaler = joblib.load(MODELS_DIR / "autoencoder_scaler.pkl")

    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
    
    X_pred = autoencoder.predict(X_scaled, batch_size=AUTOENCODER_CONFIG["batch_size"])
    reconstruction_error = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

    return reconstruction_error
