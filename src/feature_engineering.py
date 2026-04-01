"""
Feature Engineering for the IEEE Fraud Detection Dataset.
Creates behavioral, velocity, risk-based, and aggregate statistical features
to improve fraud detection accuracy.
"""

import numpy as np
import pandas as pd
from typing import List
from src.config import TIME_WINDOWS
from src.utils import timer, logger


@timer
def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral features based on spending patterns per card.
    
    These features capture how a transaction compares to a card's normal
    behavior — fraudulent transactions often deviate significantly from
    a card's historical spending patterns.
    
    Features created:
        - amt_mean_by_card: Average transaction amount per card
        - amt_std_by_card: Standard deviation of amounts per card
        - amt_median_by_card: Median amount per card
        - amt_max_by_card: Maximum amount per card
        - amt_ratio_to_mean: Current amount / card's average (anomaly signal)
        - amt_zscore: Z-score of current amount relative to card's distribution
        - amt_deviation_from_median: Absolute deviation from card's median
    """
    logger.info("  Creating behavioral features...")

    # Group by card1 (primary card identifier)
    card_stats = df.groupby("card1")["TransactionAmt"].agg(["mean", "std", "median", "max", "min", "count"])
    card_stats.columns = [
        "amt_mean_by_card", "amt_std_by_card", "amt_median_by_card",
        "amt_max_by_card", "amt_min_by_card", "txn_count_by_card"
    ]

    df = df.merge(card_stats, on="card1", how="left", suffixes=("", "_card_stat"))

    # Ratio features: how unusual is this transaction for the card?
    df["amt_ratio_to_mean"] = df["TransactionAmt"] / (df["amt_mean_by_card"] + 1e-6)
    df["amt_ratio_to_max"] = df["TransactionAmt"] / (df["amt_max_by_card"] + 1e-6)
    
    # Z-score: number of standard deviations from the mean
    df["amt_zscore"] = (
        (df["TransactionAmt"] - df["amt_mean_by_card"]) / 
        (df["amt_std_by_card"] + 1e-6)
    )
    
    # Deviation from median (more robust to outliers)
    df["amt_deviation_from_median"] = np.abs(
        df["TransactionAmt"] - df["amt_median_by_card"]
    )

    # Is this an unusually large transaction for this card?
    df["is_high_amount"] = (df["amt_ratio_to_mean"] > 3).astype(np.int8)

    logger.info(f"    Created 10 behavioral features")
    return df


@timer
def create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create velocity features: transaction frequency in time windows.
    
    Rapid succession of transactions is a strong fraud signal.
    Fraudsters often make multiple transactions quickly before being detected.
    
    Features created per time window (1h, 6h, 24h, 7d):
        - txn_count_{window}: Number of transactions by card in window
        - amt_sum_{window}: Total amount spent by card in window
    """
    logger.info("  Creating velocity features...")

    # Sort by card and time for efficient windowed computation
    df = df.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

    for window_name, window_seconds in TIME_WINDOWS.items():
        count_col = f"txn_count_{window_name}"
        amt_col = f"amt_sum_{window_name}"

        # Initialize columns
        df[count_col] = 0
        df[amt_col] = 0.0

        # For each card, count transactions within the time window
        # Using a vectorized approach with groupby + rolling for efficiency
        for card_id, group in df.groupby("card1"):
            idx = group.index
            times = group["TransactionDT"].values
            amounts = group["TransactionAmt"].values

            counts = np.zeros(len(group), dtype=np.int32)
            amt_sums = np.zeros(len(group), dtype=np.float32)

            for i in range(len(group)):
                # Look back within the time window
                mask = (times[i] - times[:i+1]) <= window_seconds
                counts[i] = mask.sum()
                amt_sums[i] = amounts[:i+1][mask].sum()

            df.loc[idx, count_col] = counts
            df.loc[idx, amt_col] = amt_sums

        logger.info(f"    Velocity window {window_name}: computed")

    # Transaction frequency acceleration: is the rate increasing?
    if "txn_count_1h" in df.columns and "txn_count_24h" in df.columns:
        df["velocity_ratio_1h_24h"] = (
            df["txn_count_1h"] / (df["txn_count_24h"] + 1e-6)
        )

    logger.info(f"    Created {len(TIME_WINDOWS) * 2 + 1} velocity features")
    return df


@timer
def create_velocity_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast approximation of velocity features using rolling windows.
    Used instead of the exact method for large datasets to save computation time.
    
    Approximates transaction velocity per card using sorted time-based grouping.
    """
    logger.info("  Creating velocity features (fast mode)...")

    df = df.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

    # Use time-binned approach for efficiency
    for window_name, window_seconds in TIME_WINDOWS.items():
        # Create time bins
        bin_col = f"time_bin_{window_name}"
        df[bin_col] = df["TransactionDT"] // window_seconds

        # Count transactions per card per time bin
        count_col = f"txn_count_{window_name}"
        amt_col = f"amt_sum_{window_name}"
        
        group_stats = df.groupby(["card1", bin_col]).agg(
            txn_count=("TransactionAmt", "count"),
            amt_sum=("TransactionAmt", "sum")
        ).reset_index()
        group_stats.columns = ["card1", bin_col, count_col, amt_col]

        # Merge back
        df = df.merge(group_stats, on=["card1", bin_col], how="left", suffixes=("", f"_{window_name}_merge"))
        
        # Drop temporary bin column
        df = df.drop(columns=[bin_col])

    # Velocity ratio: short-term vs long-term activity
    if "txn_count_1h" in df.columns and "txn_count_24h" in df.columns:
        df["velocity_ratio_1h_24h"] = df["txn_count_1h"] / (df["txn_count_24h"] + 1e-6)
    
    if "txn_count_1h" in df.columns and "txn_count_7d" in df.columns:
        df["velocity_ratio_1h_7d"] = df["txn_count_1h"] / (df["txn_count_7d"] + 1e-6)

    logger.info(f"    Created {len(TIME_WINDOWS) * 2 + 2} velocity features")
    return df


@timer
def create_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create risk-based features that capture suspicious patterns.
    
    These features model risk signals that domain experts use:
    - Geographic mismatches (billing vs shipping address changes)
    - Device changes (new or different device usage)
    - Email domain risk (some domains have higher fraud rates)
    - Time-of-day risk (fraud patterns vary by hour)
    
    Features created:
        - email_domain_risk: Historical fraud rate per email domain
        - addr_change: Whether billing address differs from card's primary
        - device_risk_score: Risk based on device type + info
        - hour_of_day: Extracted hour (fraud peaks at certain hours)
        - is_night_txn: Transaction between midnight and 6 AM
        - is_weekend: Transaction on weekend
    """
    logger.info("  Creating risk-based features...")

    # === Email Domain Risk ===
    # Calculate fraud rate per email domain (P_emaildomain)
    if "P_emaildomain" in df.columns:
        email_fraud_rate = df.groupby("P_emaildomain")["isFraud"].mean()
        df["email_domain_risk"] = df["P_emaildomain"].map(email_fraud_rate).fillna(
            df["isFraud"].mean()  # Default to overall fraud rate
        )
    
    if "R_emaildomain" in df.columns:
        r_email_fraud_rate = df.groupby("R_emaildomain")["isFraud"].mean()
        df["r_email_domain_risk"] = df["R_emaildomain"].map(r_email_fraud_rate).fillna(
            df["isFraud"].mean()
        )

    # === Email Match ===
    # Check if purchaser and recipient email domains match
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(np.int8)

    # === Address Change Detection ===
    # Track if address is different from card's most common address
    if "addr1" in df.columns:
        card_addr_mode = df.groupby("card1")["addr1"].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        df["primary_addr"] = df["card1"].map(card_addr_mode)
        df["addr_change"] = (df["addr1"] != df["primary_addr"]).astype(np.int8)
        df = df.drop(columns=["primary_addr"])

    # === Device Risk ===
    if "DeviceType" in df.columns:
        device_fraud_rate = df.groupby("DeviceType")["isFraud"].mean()
        df["device_type_risk"] = df["DeviceType"].map(device_fraud_rate).fillna(
            df["isFraud"].mean()
        )

    if "DeviceInfo" in df.columns:
        # Track if device is new for this card
        card_device_mode = df.groupby("card1")["DeviceInfo"].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        df["primary_device"] = df["card1"].map(card_device_mode)
        df["device_change"] = (df["DeviceInfo"] != df["primary_device"]).astype(np.int8)
        df = df.drop(columns=["primary_device"])

    # === Temporal Risk Features ===
    # TransactionDT appears to be seconds from some reference point
    # Extract useful time components
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    df["is_night_txn"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(np.int8)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    # === Transaction Amount Risk ===
    # Round amounts are suspicious (e.g., exactly $100, $500)
    df["is_round_amount"] = (df["TransactionAmt"] % 1 == 0).astype(np.int8)
    df["log_amount"] = np.log1p(df["TransactionAmt"])

    # === Product Code Risk ===
    if "ProductCD" in df.columns:
        product_fraud_rate = df.groupby("ProductCD")["isFraud"].mean()
        df["product_risk"] = df["ProductCD"].map(product_fraud_rate).fillna(
            df["isFraud"].mean()
        )

    logger.info(f"    Created risk-based features")
    return df


@timer
def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate historical statistics per card.
    
    These features capture long-term card behavior patterns:
    - Total transaction history
    - Average/max spending levels
    - Time-based aggregations
    
    Features created:
        - card_total_txn: Total historical transactions per card
        - card_avg_amt: Average amount per card
        - card_max_amt: Maximum amount per card
        - card_amt_range: Max - Min amount (spending range)
        - days_since_first_txn: Days since card's first transaction
        - card_unique_addr: Number of unique addresses per card
    """
    logger.info("  Creating aggregate features...")

    # === Per-card aggregate stats ===
    card_agg = df.groupby("card1").agg(
        card_total_txn=("TransactionAmt", "count"),
        card_avg_amt=("TransactionAmt", "mean"),
        card_max_amt=("TransactionAmt", "max"),
        card_min_amt=("TransactionAmt", "min"),
        card_first_txn_dt=("TransactionDT", "min"),
        card_last_txn_dt=("TransactionDT", "max"),
    )
    
    card_agg["card_amt_range"] = card_agg["card_max_amt"] - card_agg["card_min_amt"]
    card_agg["card_active_days"] = (card_agg["card_last_txn_dt"] - card_agg["card_first_txn_dt"]) / 86400

    df = df.merge(card_agg, on="card1", how="left", suffixes=("", "_agg"))
    
    # Days since first transaction for this card
    df["days_since_first_txn"] = (df["TransactionDT"] - df["card_first_txn_dt"]) / 86400
    
    # Clean up temporary columns
    df = df.drop(columns=["card_first_txn_dt", "card_last_txn_dt"], errors="ignore")

    # === Per-card unique counts ===
    if "addr1" in df.columns:
        addr_counts = df.groupby("card1")["addr1"].nunique().rename("card_unique_addr")
        df = df.merge(addr_counts, on="card1", how="left")

    if "DeviceInfo" in df.columns:
        device_counts = df.groupby("card1")["DeviceInfo"].nunique().rename("card_unique_devices")
        df = df.merge(device_counts, on="card1", how="left")

    # === Interaction Features ===
    # Card + Address combination (captures card-address pairs)
    if "addr1" in df.columns:
        df["card_addr_interaction"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        card_addr_counts = df["card_addr_interaction"].map(df["card_addr_interaction"].value_counts())
        df["card_addr_frequency"] = card_addr_counts
        df = df.drop(columns=["card_addr_interaction"])

    logger.info(f"    Created aggregate features")
    return df


@timer
def run_feature_engineering(df: pd.DataFrame, fast_mode: bool = True) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame.
        fast_mode: If True, use fast approximation for velocity features.
    
    Returns:
        DataFrame with all engineered features.
    """
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)

    initial_features = len(df.columns)

    df = create_behavioral_features(df)
    
    if fast_mode:
        df = create_velocity_features_fast(df)
    else:
        df = create_velocity_features(df)
    
    df = create_risk_features(df)
    df = create_aggregate_features(df)

    final_features = len(df.columns)
    logger.info(f"  Feature engineering complete: {initial_features} -> {final_features} features")
    logger.info(f"  New features created: {final_features - initial_features}")

    return df
