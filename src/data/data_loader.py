"""
Data loader: downloads OHLCV data via yfinance and returns a clean DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config.settings import (
    DEFAULT_TICKER,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_INTERVAL,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)


def load_ohlcv(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    interval: str = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download or load from cache OHLCV data for *ticker*.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume  — indexed by UTC date.
    """
    cache_path = CACHE_DIR / f"{ticker}_{start}_{end}_{interval}.parquet"

    if use_cache and cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        df = pd.read_parquet(cache_path)
    else:
        logger.info("Downloading %s from %s to %s …", ticker, start, end)
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. Check the symbol and date range."
            )
        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        if use_cache:
            df.to_parquet(cache_path)
            logger.info("Cached data saved to %s", cache_path)

    return df


def load_from_csv(filepath: str | Path) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file. Expects a 'Date' or datetime index."""
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df[list(required)]
