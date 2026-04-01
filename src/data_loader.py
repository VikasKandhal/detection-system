"""
Data Loader for the IEEE Fraud Detection Dataset.
Loads, merges, and memory-optimizes the transaction + identity datasets.
"""

import pandas as pd
from src.config import (
    TRAIN_TRANSACTION, TRAIN_IDENTITY,
    TEST_TRANSACTION, TEST_IDENTITY,
    PROCESSED_DATA_DIR
)
from src.utils import reduce_memory_usage, timer, logger, save_dataframe


@timer
def load_and_merge_data(
    transaction_path: str = None,
    identity_path: str = None,
    is_train: bool = True,
    nrows: int = None
) -> pd.DataFrame:
    """
    Load and merge transaction + identity data.
    
    Uses left join on TransactionID since not all transactions have identity info.
    Identity data is available for ~25% of transactions (those with additional
    identity verification like device fingerprinting).
    
    Args:
        transaction_path: Path to transaction CSV. Defaults to train file.
        identity_path: Path to identity CSV. Defaults to train file.
        is_train: Whether loading training data (has isFraud column).
        nrows: Number of rows to load (for debugging). None = all rows.
    
    Returns:
        Merged DataFrame with memory-optimized dtypes.
    """
    if transaction_path is None:
        transaction_path = TRAIN_TRANSACTION if is_train else TEST_TRANSACTION
    if identity_path is None:
        identity_path = TRAIN_IDENTITY if is_train else TEST_IDENTITY

    logger.info(f"Loading transactions from: {transaction_path}")
    df_transaction = pd.read_csv(transaction_path, nrows=nrows)
    logger.info(f"  Transaction shape: {df_transaction.shape}")

    logger.info(f"Loading identity from: {identity_path}")
    df_identity = pd.read_csv(identity_path, nrows=nrows)
    logger.info(f"  Identity shape: {df_identity.shape}")

    # Left join: keep all transactions, add identity info where available
    logger.info("Merging transaction + identity on TransactionID...")
    df = df_transaction.merge(df_identity, on="TransactionID", how="left")
    logger.info(f"  Merged shape: {df.shape}")

    # Report identity match rate
    id_cols = [c for c in df_identity.columns if c != "TransactionID"]
    if id_cols:
        match_rate = df[id_cols[0]].notna().mean() * 100
        logger.info(f"  Identity match rate: {match_rate:.1f}%")

    # Reduce memory usage via dtype downcasting
    df = reduce_memory_usage(df)

    return df


@timer
def load_processed_data(name: str = "train_merged") -> pd.DataFrame:
    """Load previously processed and saved data from parquet."""
    from src.utils import load_dataframe
    return load_dataframe(PROCESSED_DATA_DIR, name)


@timer
def save_processed_data(df: pd.DataFrame, name: str = "train_merged"):
    """Save processed data to parquet for fast reloading."""
    save_dataframe(df, PROCESSED_DATA_DIR, name)


if __name__ == "__main__":
    # Quick test: load a small sample
    df = load_and_merge_data(nrows=10000)
    print(f"\nSample data shape: {df.shape}")
    print(f"Columns: {list(df.columns[:20])}...")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")
