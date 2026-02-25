"""
Utility helpers: logging setup, report export, and general-purpose functions.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean, informative format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs(*paths: str) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_metrics_csv(
    metrics: dict, filepath: str = "reports/performance_summary.csv"
) -> None:
    """Persist the metrics dictionary to a CSV file."""
    ensure_dirs(str(Path(filepath).parent))
    df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    df.index.name = "Metric"
    df.to_csv(filepath)
    logging.getLogger(__name__).info("Metrics saved to %s", filepath)


def save_trades_csv(
    trades_df: pd.DataFrame, filepath: str = "reports/trades.csv"
) -> None:
    """Persist the trade log to a CSV file."""
    ensure_dirs(str(Path(filepath).parent))
    trades_df.to_csv(filepath, index=False)
    logging.getLogger(__name__).info("Trade log saved to %s", filepath)


def print_metrics_table(metrics: dict) -> None:
    """Print a formatted metrics table to stdout."""
    width = 42
    print("\n" + "═" * width)
    print(f"{'PERFORMANCE SUMMARY':^{width}}")
    print("═" * width)
    for key, value in metrics.items():
        print(f"  {key:<28} {value:>10}")
    print("═" * width + "\n")


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value:+.2f}%"
