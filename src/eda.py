"""
Exploratory Data Analysis (EDA) for the IEEE Fraud Detection Dataset.
Generates comprehensive visualizations for fraud distribution, feature analysis,
missing values, temporal patterns, and transaction amounts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import FIGURES_DIR, CATEGORICAL_FEATURES
from src.utils import timer, logger


# Set global plot style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox_inches": "tight"
})


@timer
def run_full_eda(df: pd.DataFrame, save_dir: Path = None):
    """
    Run complete EDA pipeline and save all plots.
    
    Args:
        df: Merged training DataFrame.
        save_dir: Directory to save figures. Defaults to FIGURES_DIR.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running EDA on {len(df)} rows, {len(df.columns)} columns")
    
    plot_fraud_distribution(df, save_dir)
    plot_transaction_amounts(df, save_dir)
    plot_missing_values(df, save_dir)
    plot_temporal_analysis(df, save_dir)
    plot_categorical_distributions(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    print_dataset_summary(df)

    logger.info(f"EDA complete. All plots saved to {save_dir}")


def plot_fraud_distribution(df: pd.DataFrame, save_dir: Path):
    """Plot fraud vs legitimate transaction distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    fraud_counts = df["isFraud"].value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0, 0.1)
    
    axes[0].pie(
        fraud_counts.values, labels=labels, colors=colors,
        explode=explode, autopct="%1.2f%%", shadow=True,
        startangle=140, textprops={"fontsize": 13, "fontweight": "bold"}
    )
    axes[0].set_title("Fraud Distribution", fontsize=16, fontweight="bold")

    # Bar chart with counts
    bars = axes[1].bar(labels, fraud_counts.values, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Transaction Counts", fontsize=16, fontweight="bold")
    axes[1].set_ylabel("Count")
    
    # Add count labels on bars
    for bar, count in zip(bars, fraud_counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f"{count:,}", ha="center", fontsize=13, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_dir / "01_fraud_distribution.png")
    plt.close()
    logger.info("  Saved: fraud_distribution.png")


def plot_transaction_amounts(df: pd.DataFrame, save_dir: Path):
    """Plot transaction amount distributions for fraud vs legitimate."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distribution plot
    fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"]
    legit_amt = df[df["isFraud"] == 0]["TransactionAmt"]

    # Log-scale histogram (amounts are highly skewed)
    axes[0].hist(
        np.log1p(legit_amt), bins=100, alpha=0.7, label="Legitimate",
        color="#2ecc71", density=True
    )
    axes[0].hist(
        np.log1p(fraud_amt), bins=100, alpha=0.7, label="Fraud",
        color="#e74c3c", density=True
    )
    axes[0].set_xlabel("log(TransactionAmt + 1)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Transaction Amount Distribution (Log Scale)", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=12)

    # Box plot comparison  
    df_plot = df[["TransactionAmt", "isFraud"]].copy()
    df_plot["TransactionAmt_log"] = np.log1p(df_plot["TransactionAmt"])
    df_plot["Class"] = df_plot["isFraud"].map({0: "Legitimate", 1: "Fraud"})
    
    sns.boxplot(data=df_plot, x="Class", y="TransactionAmt_log", 
                palette={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"}, ax=axes[1])
    axes[1].set_ylabel("log(TransactionAmt + 1)")
    axes[1].set_title("Amount by Class (Log Scale)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "02_transaction_amounts.png")
    plt.close()
    logger.info("  Saved: transaction_amounts.png")


def plot_missing_values(df: pd.DataFrame, save_dir: Path):
    """Plot top features with highest missing value percentages."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(40)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ["#e74c3c" if v > 80 else "#f39c12" if v > 50 else "#3498db" for v in missing_pct.values]
    
    bars = ax.barh(range(len(missing_pct)), missing_pct.values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(missing_pct)))
    ax.set_yticklabels(missing_pct.index, fontsize=9)
    ax.set_xlabel("Missing %")
    ax.set_title("Top 40 Features by Missing Value Percentage", fontsize=16, fontweight="bold")
    ax.axvline(x=80, color="red", linestyle="--", alpha=0.7, label="80% threshold (drop)")
    ax.legend(fontsize=11)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_dir / "03_missing_values.png")
    plt.close()
    logger.info("  Saved: missing_values.png")


def plot_temporal_analysis(df: pd.DataFrame, save_dir: Path):
    """Analyze fraud rate over time using TransactionDT."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Convert TransactionDT to days (it's in seconds from some reference point)
    df_temp = df[["TransactionDT", "isFraud"]].copy()
    df_temp["day"] = df_temp["TransactionDT"] // 86400

    # Daily fraud rate
    daily = df_temp.groupby("day").agg(
        total=("isFraud", "count"),
        fraud=("isFraud", "sum")
    )
    daily["fraud_rate"] = daily["fraud"] / daily["total"] * 100

    axes[0].plot(daily.index, daily["fraud_rate"], color="#e74c3c", alpha=0.8, linewidth=1.5)
    axes[0].fill_between(daily.index, daily["fraud_rate"], alpha=0.2, color="#e74c3c")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Fraud Rate (%)")
    axes[0].set_title("Daily Fraud Rate Over Time", fontsize=14, fontweight="bold")
    axes[0].axhline(y=df["isFraud"].mean()*100, color="blue", linestyle="--", alpha=0.5, label="Overall avg")
    axes[0].legend()

    # Daily transaction volume
    axes[1].bar(daily.index, daily["total"], color="#3498db", alpha=0.7, label="Total")
    axes[1].bar(daily.index, daily["fraud"], color="#e74c3c", alpha=0.8, label="Fraud")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Transaction Count")
    axes[1].set_title("Daily Transaction Volume", fontsize=14, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "04_temporal_analysis.png")
    plt.close()
    logger.info("  Saved: temporal_analysis.png")


def plot_categorical_distributions(df: pd.DataFrame, save_dir: Path):
    """Plot fraud rates across key categorical features."""
    key_cats = ["ProductCD", "card4", "card6", "DeviceType"]
    available_cats = [c for c in key_cats if c in df.columns]

    if not available_cats:
        logger.warning("No key categorical features found for plotting")
        return

    n = len(available_cats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, available_cats):
        fraud_rate = df.groupby(col)["isFraud"].mean().sort_values(ascending=False).head(10)
        bars = ax.bar(range(len(fraud_rate)), fraud_rate.values * 100, 
                      color=sns.color_palette("RdYlGn_r", len(fraud_rate)), edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(fraud_rate)))
        ax.set_xticklabels(fraud_rate.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title(f"Fraud Rate by {col}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "05_categorical_fraud_rates.png")
    plt.close()
    logger.info("  Saved: categorical_fraud_rates.png")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Path):
    """Plot correlation heatmap of the most fraud-correlated numeric features."""
    # Select numeric columns and compute correlation with isFraud
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if "isFraud" not in numeric_cols:
        logger.warning("isFraud not found in numeric columns")
        return

    correlations = df[numeric_cols].corr()["isFraud"].drop("isFraud").abs().sort_values(ascending=False)
    top_features = correlations.head(20).index.tolist()
    top_features = ["isFraud"] + top_features

    fig, ax = plt.subplots(figsize=(14, 12))
    corr_matrix = df[top_features].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        linewidths=0.5, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Correlation Heatmap (Top 20 Fraud-Correlated Features)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "06_correlation_heatmap.png")
    plt.close()
    logger.info("  Saved: correlation_heatmap.png")


def print_dataset_summary(df: pd.DataFrame):
    """Print a comprehensive dataset summary."""
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Total transactions: {len(df):,}")
    logger.info(f"  Fraud transactions: {df['isFraud'].sum():,}")
    logger.info(f"  Legitimate transactions: {(1 - df['isFraud']).sum():,.0f}")
    logger.info(f"  Fraud rate: {df['isFraud'].mean() * 100:.2f}%")
    logger.info(f"  Numeric features: {len(df.select_dtypes(include=[np.number]).columns)}")
    logger.info(f"  Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
    logger.info(f"  Total missing values: {df.isnull().sum().sum():,}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    logger.info("=" * 70)
