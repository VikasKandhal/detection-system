"""
Run the EDA analysis standalone.
Usage: python scripts/run_eda.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_and_merge_data
from src.eda import run_full_eda
from src.utils import logger


def main():
    logger.info("Running EDA Analysis...")
    df = load_and_merge_data()
    run_full_eda(df)
    logger.info("EDA complete! Check reports/figures/ for plots.")


if __name__ == "__main__":
    main()
