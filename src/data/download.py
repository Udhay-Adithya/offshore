"""
Data download utilities for Offshore.

Handles fetching historical OHLCV data from various sources.
Currently supports Yahoo Finance via yfinance.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Download historical stock data from various sources.

    Currently supports:
    - Yahoo Finance (via yfinance)

    The design allows easy extension to other data sources
    (broker APIs, alternative data providers, etc.).

    Example:
        >>> downloader = DataDownloader(output_dir="data/raw")
        >>> df = downloader.download("AAPL", "2020-01-01", "2024-01-01", "1d")
    """

    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"]

    def __init__(self, output_dir: Union[str, Path] = "data/raw", source: str = "yfinance"):
        """
        Initialize the data downloader.

        Args:
            output_dir: Directory to save downloaded data.
            source: Data source to use. Currently only "yfinance" supported.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source = source

        if source != "yfinance":
            raise NotImplementedError(f"Data source '{source}' not implemented. Use 'yfinance'.")

    def download(
        self, ticker: str, start_date: str, end_date: str, interval: str = "1d", save: bool = True
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data for a stock.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL").
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            interval: Data interval/frequency. Valid values:
                     "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                     "1d", "5d", "1wk", "1mo"
            save: Whether to save the data to a CSV file.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            Index: DatetimeIndex

        Raises:
            ValueError: If invalid interval or no data returned.
            ImportError: If yfinance not installed.
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. " f"Valid intervals: {self.VALID_INTERVALS}"
            )

        logger.info(f"Downloading {ticker} data from {start_date} to {end_date} ({interval})")

        df = self._download_yfinance(ticker, start_date, end_date, interval)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}. Check ticker symbol and date range.")

        # Standardize column names
        df = self._standardize_columns(df)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        logger.info(f"Downloaded {len(df)} rows for {ticker}")

        if save:
            filepath = self._save_data(df, ticker, interval)
            logger.info(f"Saved data to {filepath}")

        return df

    def _download_yfinance(
        self, ticker: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame:
        """Download data using yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for downloading data. "
                "Install it with: pip install yfinance"
            )

        # Create ticker object
        stock = yf.Ticker(ticker)

        # Download historical data
        df = stock.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,  # Keep both Close and Adj Close
        )

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }

        df = df.rename(columns=column_mapping)

        # Keep only essential columns
        essential_cols = ["open", "high", "low", "close", "volume", "adj_close"]
        available_cols = [col for col in essential_cols if col in df.columns]

        return df[available_cols]

    def _save_data(self, df: pd.DataFrame, ticker: str, interval: str) -> Path:
        """Save data to CSV file."""
        filename = f"{ticker}_{interval}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath)
        return filepath

    def load(self, ticker: str, interval: str = "1d") -> pd.DataFrame:
        """
        Load previously downloaded data.

        Args:
            ticker: Stock ticker symbol.
            interval: Data interval.

        Returns:
            DataFrame with OHLCV data.

        Raises:
            FileNotFoundError: If data file doesn't exist.
        """
        filename = f"{ticker}_{interval}.csv"
        filepath = self.output_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}. " f"Download it first with download() method."
            )

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df


def download_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    output_dir: Union[str, Path] = "data/raw",
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a stock.

    Convenience function that creates a DataDownloader and downloads data.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        interval: Data interval (default "1d").
        output_dir: Directory to save data (default "data/raw").
        save: Whether to save to CSV (default True).

    Returns:
        DataFrame with OHLCV data.

    Example:
        >>> df = download_data("AAPL", "2020-01-01", "2024-01-01")
        >>> df.head()
    """
    downloader = DataDownloader(output_dir=output_dir)
    return downloader.download(ticker, start_date, end_date, interval, save)
