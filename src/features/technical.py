"""
Technical indicator calculations for Offshore.

Implements common technical indicators for stock price analysis.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Calculate technical indicators for stock data.

    All methods are static and return the DataFrame with new columns added.

    Example:
        >>> ti = TechnicalIndicators()
        >>> df = ti.add_sma(df, window=20)
        >>> df = ti.add_rsi(df, window=14)
    """

    @staticmethod
    def returns(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
        """Calculate simple returns."""
        df = df.copy()
        df["returns"] = df[col].pct_change()
        return df

    @staticmethod
    def log_returns(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
        """Calculate log returns."""
        df = df.copy()
        df["log_returns"] = np.log(df[col] / df[col].shift(1))
        return df

    @staticmethod
    def volatility(df: pd.DataFrame, window: int = 20, col: str = "close") -> pd.DataFrame:
        """Calculate rolling volatility (standard deviation of returns)."""
        df = df.copy()
        if "returns" not in df.columns:
            df["returns"] = df[col].pct_change()
        df["volatility"] = df["returns"].rolling(window=window).std()
        return df

    @staticmethod
    def sma(
        df: pd.DataFrame, window: int, col: str = "close", suffix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate Simple Moving Average.

        Args:
            df: DataFrame with price data.
            window: Moving average window size.
            col: Column to calculate SMA on.
            suffix: Optional suffix for column name. If None, uses window.
        """
        df = df.copy()
        col_name = f"sma_{suffix or window}"
        df[col_name] = df[col].rolling(window=window).mean()
        return df

    @staticmethod
    def ema(
        df: pd.DataFrame, span: int, col: str = "close", suffix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with price data.
            span: EMA span.
            col: Column to calculate EMA on.
            suffix: Optional suffix for column name.
        """
        df = df.copy()
        col_name = f"ema_{suffix or span}"
        df[col_name] = df[col].ewm(span=span, adjust=False).mean()
        return df

    @staticmethod
    def rsi(df: pd.DataFrame, window: int = 14, col: str = "close") -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with price data.
            window: RSI lookback period.
            col: Column to calculate RSI on.
        """
        df = df.copy()

        # Calculate price changes
        delta = df[col].diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Calculate average gain and loss using EMA
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df[f"rsi_{window}"] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with price data.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line EMA period.
            col: Column to calculate MACD on.
        """
        df = df.copy()

        # Calculate EMAs
        ema_fast = df[col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[col].ewm(span=slow, adjust=False).mean()

        # MACD line
        df["macd"] = ema_fast - ema_slow

        # Signal line
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()

        # MACD histogram
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, window: int = 20, num_std: float = 2.0, col: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with price data.
            window: Moving average window.
            num_std: Number of standard deviations for bands.
            col: Column to calculate on.
        """
        df = df.copy()

        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std = df[col].rolling(window=window).std()

        df["bollinger_middle"] = rolling_mean
        df["bollinger_upper"] = rolling_mean + (rolling_std * num_std)
        df["bollinger_lower"] = rolling_mean - (rolling_std * num_std)

        # Normalized position within bands
        df["bollinger_pct"] = (df[col] - df["bollinger_lower"]) / (
            df["bollinger_upper"] - df["bollinger_lower"]
        )

        return df

    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        Args:
            df: DataFrame with OHLC data.
            window: ATR lookback period.
        """
        df = df.copy()

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        df[f"atr_{window}"] = true_range.rolling(window=window).mean()

        return df

    @staticmethod
    def volume_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate volume Simple Moving Average."""
        df = df.copy()
        df[f"volume_sma_{window}"] = df["volume"].rolling(window=window).mean()

        # Relative volume
        df["relative_volume"] = df["volume"] / df[f"volume_sma_{window}"]

        return df

    @staticmethod
    def price_to_sma(df: pd.DataFrame, window: int = 20, col: str = "close") -> pd.DataFrame:
        """Calculate price relative to SMA (normalized)."""
        df = df.copy()

        sma_col = f"sma_{window}"
        if sma_col not in df.columns:
            df[sma_col] = df[col].rolling(window=window).mean()

        df[f"price_to_sma_{window}"] = (df[col] - df[sma_col]) / df[sma_col]

        return df

    @staticmethod
    def high_low_range(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate high-low range normalized by close."""
        df = df.copy()
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        return df

    @staticmethod
    def momentum(df: pd.DataFrame, window: int = 10, col: str = "close") -> pd.DataFrame:
        """Calculate price momentum (rate of change)."""
        df = df.copy()
        df[f"momentum_{window}"] = df[col].pct_change(periods=window)
        return df


def add_technical_indicators(
    df: pd.DataFrame, indicators: list[str], close_col: str = "close"
) -> pd.DataFrame:
    """
    Add technical indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data.
        indicators: List of indicator names to add. Supported:
            - returns, log_returns, volatility
            - sma_5, sma_10, sma_20, sma_50 (or sma_N for any N)
            - ema_12, ema_26 (or ema_N for any N)
            - rsi_14 (or rsi_N for any N)
            - macd, macd_signal, macd_hist
            - bollinger_upper, bollinger_lower, bollinger_pct
            - atr_14 (or atr_N for any N)
            - volume_sma_20 (or volume_sma_N for any N)
            - price_to_sma_20 (or price_to_sma_N for any N)
            - high_low_range
            - momentum_10 (or momentum_N for any N)
        close_col: Name of the close price column.

    Returns:
        DataFrame with added indicator columns.
    """
    df = df.copy()
    ti = TechnicalIndicators()

    # Track which indicators have been added
    added: set[str] = set()

    for indicator in indicators:
        indicator = indicator.lower()

        # Parse indicator name and window if applicable
        parts = indicator.rsplit("_", 1)
        base_name = parts[0]
        window = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

        try:
            if indicator == "returns" and indicator not in added:
                df = ti.returns(df, close_col)
                added.add("returns")

            elif indicator == "log_returns" and indicator not in added:
                df = ti.log_returns(df, close_col)
                added.add("log_returns")

            elif indicator == "volatility" and indicator not in added:
                df = ti.volatility(df, window=20, col=close_col)
                added.add("volatility")

            elif base_name == "sma" and window:
                col_name = f"sma_{window}"
                if col_name not in added:
                    df = ti.sma(df, window=window, col=close_col)
                    added.add(col_name)

            elif base_name == "ema" and window:
                col_name = f"ema_{window}"
                if col_name not in added:
                    df = ti.ema(df, span=window, col=close_col)
                    added.add(col_name)

            elif base_name == "rsi":
                w = window or 14
                col_name = f"rsi_{w}"
                if col_name not in added:
                    df = ti.rsi(df, window=w, col=close_col)
                    added.add(col_name)

            elif indicator in ["macd", "macd_signal", "macd_hist"]:
                if "macd" not in added:
                    df = ti.macd(df, col=close_col)
                    added.update(["macd", "macd_signal", "macd_hist"])

            elif indicator in ["bollinger_upper", "bollinger_lower", "bollinger_pct"]:
                if "bollinger_upper" not in added:
                    df = ti.bollinger_bands(df, col=close_col)
                    added.update(
                        ["bollinger_middle", "bollinger_upper", "bollinger_lower", "bollinger_pct"]
                    )

            elif base_name == "atr":
                w = window or 14
                col_name = f"atr_{w}"
                if col_name not in added:
                    df = ti.atr(df, window=w)
                    added.add(col_name)

            elif base_name == "volume_sma":
                w = window or 20
                col_name = f"volume_sma_{w}"
                if col_name not in added:
                    df = ti.volume_sma(df, window=w)
                    added.update([col_name, "relative_volume"])

            elif base_name == "price_to_sma":
                w = window or 20
                col_name = f"price_to_sma_{w}"
                if col_name not in added:
                    df = ti.price_to_sma(df, window=w, col=close_col)
                    added.add(col_name)

            elif indicator == "high_low_range" and indicator not in added:
                df = ti.high_low_range(df)
                added.add("high_low_range")

            elif base_name == "momentum":
                w = window or 10
                col_name = f"momentum_{w}"
                if col_name not in added:
                    df = ti.momentum(df, window=w, col=close_col)
                    added.add(col_name)

        except Exception as e:
            import logging

            logging.warning(f"Failed to add indicator {indicator}: {e}")

    return df
