"""
Utility functions for the Fraud Detection System.
Provides memory optimization, logging, timing, and common helpers.
"""

import time
import logging
import functools
import numpy as np
import pandas as pd
from pathlib import Path


def setup_logging(name: str = "fraud_detection", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logging()


def timer(func):
    """Decorator to log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Starting: {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        minutes, seconds = divmod(elapsed, 60)
        if minutes > 0:
            logger.info(f"Completed: {func.__name__} in {int(minutes)}m {seconds:.1f}s")
        else:
            logger.info(f"Completed: {func.__name__} in {seconds:.2f}s")
        return result
    return wrapper


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True, protected_cols: list = None) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    This is critical for handling the large IEEE dataset efficiently.
    
    Args:
        df: Input DataFrame.
        verbose: If True, print memory savings.
        protected_cols: Columns to skip downcasting (merge/groupby keys).
    
    Returns:
        Memory-optimized DataFrame.
    """
    if protected_cols is None:
        # Protect columns commonly used as merge keys to avoid
        # pandas Buffer dtype mismatch errors with int8/int16
        protected_cols = {"TransactionID", "TransactionDT", "card1", "card2", "card3", "card5", "addr1", "addr2", "isFraud"}

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        if col in protected_cols:
            continue

        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            # Downcast integers
            if str(col_type).startswith("int"):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            # Downcast floats
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(
            f"Memory reduced from {start_mem:.1f} MB to {end_mem:.1f} MB "
            f"({reduction:.1f}% reduction)"
        )

    return df


def save_dataframe(df: pd.DataFrame, path: Path, name: str) -> Path:
    """Save a DataFrame to parquet format for efficient storage."""
    filepath = path / f"{name}.parquet"
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {name} to {filepath} ({len(df)} rows)")
    return filepath


def load_dataframe(path: Path, name: str) -> pd.DataFrame:
    """Load a DataFrame from parquet format."""
    filepath = path / f"{name}.parquet"
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {name} from {filepath} ({len(df)} rows)")
    return df
