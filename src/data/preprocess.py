"""
Data preprocessing utilities for Offshore.

Handles cleaning, feature engineering, and train/val/test splitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.features.technical import add_technical_indicators
from src.features.labeling import create_labels

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess raw OHLCV data for model training.

    Pipeline:
    1. Clean missing values
    2. Add technical indicators
    3. Create target labels
    4. Normalize features
    5. Split into train/val/test by time

    Example:
        >>> preprocessor = DataPreprocessor(config)
        >>> train_df, val_df, test_df = preprocessor.process(raw_df)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the preprocessor.

        Args:
            config: Configuration dictionary from data.yaml
        """
        self.config = config
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.feature_columns: list[str] = []

    def process(
        self, df: pd.DataFrame, fit_scaler: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process raw data through the full pipeline.

        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex.
            fit_scaler: Whether to fit the scaler (True for training data).

        Returns:
            Tuple of (train_df, val_df, test_df) with features and labels.
        """
        logger.info("Starting data preprocessing...")

        # Step 1: Clean data
        df = self._clean_data(df)
        logger.info(f"After cleaning: {len(df)} rows")

        # Step 2: Add technical indicators
        df = self._add_features(df)
        logger.info(f"Added features: {len(self.feature_columns)} columns")

        # Step 3: Create labels
        df = self._create_labels(df)
        logger.info(f"Created labels for {len(df)} samples")

        # Step 4: Split by time
        train_df, val_df, test_df = self._split_by_time(df)
        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        # Step 5: Normalize features
        if self.config.get("features", {}).get("normalize", True):
            train_df, val_df, test_df = self._normalize_features(
                train_df, val_df, test_df, fit_scaler
            )
            logger.info("Normalized features")

        return train_df, val_df, test_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values and handle data quality issues."""
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by time
        df = df.sort_index()

        # Handle missing values
        missing_config = self.config.get("missing_values", {})
        method = missing_config.get("method", "ffill")
        max_consecutive = missing_config.get("max_consecutive_missing", 5)

        # Check for excessive consecutive missing values
        if max_consecutive > 0:
            for col in df.columns:
                consecutive_nulls = (
                    df[col].isnull().astype(int).groupby((df[col].notnull()).cumsum()).cumsum()
                )

                if consecutive_nulls.max() > max_consecutive:
                    logger.warning(
                        f"Column {col} has more than {max_consecutive} "
                        f"consecutive missing values. Some may be dropped."
                    )

        # Apply missing value handling
        if method == "ffill":
            df = df.ffill()
            df = df.bfill()  # Fill any remaining at the start
        elif method == "bfill":
            df = df.bfill()
            df = df.ffill()
        elif method == "interpolate":
            df = df.interpolate(method="time")
            df = df.ffill().bfill()
        elif method == "drop":
            df = df.dropna()

        # Remove any remaining NaN rows
        df = df.dropna()

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep="first")]

        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as features."""
        features_config = self.config.get("features", {})
        indicators = features_config.get("indicators", [])

        # Add technical indicators
        df = add_technical_indicators(df, indicators)

        # Store feature columns (exclude OHLCV and target)
        exclude_cols = ["open", "high", "low", "close", "volume", "adj_close", "label"]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        # Drop rows with NaN from indicator calculations
        df = df.dropna()

        return df

    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target labels based on future returns."""
        labeling_config = self.config.get("labeling", {})

        df = create_labels(
            df=df,
            horizon=labeling_config.get("horizon", 5),
            label_type=labeling_config.get("label_type", "binary"),
            binary_threshold=labeling_config.get("binary_threshold", 0.0),
            multiclass_config=labeling_config.get("multiclass", {}),
            price_col="close",
        )

        # Drop rows without labels (at the end due to horizon)
        df = df.dropna(subset=["label"])

        # Convert label to integer
        df["label"] = df["label"].astype(int)

        return df

    def _split_by_time(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by time (no random shuffling to prevent leakage)."""
        splits_config = self.config.get("splits", {})

        train_end = pd.to_datetime(splits_config.get("train_end", "2022-06-30"))
        val_end = pd.to_datetime(splits_config.get("val_end", "2023-06-30"))

        train_df = df[df.index <= train_end].copy()
        val_df = df[(df.index > train_end) & (df.index <= val_end)].copy()
        test_df = df[df.index > val_end].copy()

        return train_df, val_df, test_df

    def _normalize_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, fit_scaler: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize features using train set statistics."""
        normalize_method = self.config.get("features", {}).get("normalize_method", "zscore")

        if fit_scaler or self.scaler is None:
            if normalize_method == "zscore":
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()

            # Fit on training data only
            self.scaler.fit(train_df[self.feature_columns])

        # Transform all sets
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df[self.feature_columns] = self.scaler.transform(train_df[self.feature_columns])
        val_df[self.feature_columns] = self.scaler.transform(val_df[self.feature_columns])
        test_df[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns])

        return train_df, val_df, test_df

    def save_processed(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ticker: str,
        output_dir: Union[str, Path] = "data/processed",
    ) -> None:
        """
        Save processed datasets to files.

        Args:
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
            ticker: Stock ticker for filename.
            output_dir: Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_dir / f"{ticker}_train.csv")
        val_df.to_csv(output_dir / f"{ticker}_val.csv")
        test_df.to_csv(output_dir / f"{ticker}_test.csv")

        # Save feature columns and scaler info
        import json

        metadata = {
            "ticker": ticker,
            "feature_columns": self.feature_columns,
            "num_features": len(self.feature_columns),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        }

        with open(output_dir / f"{ticker}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save scaler
        if self.scaler is not None:
            import joblib

            joblib.dump(self.scaler, output_dir / f"{ticker}_scaler.joblib")

        logger.info(f"Saved processed data to {output_dir}")


def preprocess_data(
    df: pd.DataFrame,
    config: dict[str, Any],
    ticker: str,
    output_dir: Union[str, Path] = "data/processed",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess raw OHLCV data for model training.

    Convenience function that creates a DataPreprocessor and processes data.

    Args:
        df: Raw OHLCV DataFrame.
        config: Configuration dictionary from data.yaml.
        ticker: Stock ticker symbol.
        output_dir: Directory to save processed data.
        save: Whether to save processed data.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    preprocessor = DataPreprocessor(config)
    train_df, val_df, test_df = preprocessor.process(df)

    if save:
        preprocessor.save_processed(train_df, val_df, test_df, ticker, output_dir)

    return train_df, val_df, test_df


def load_processed_data(
    ticker: str, data_dir: Union[str, Path] = "data/processed"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Load previously processed data.

    Args:
        ticker: Stock ticker symbol.
        data_dir: Directory with processed data.

    Returns:
        Tuple of (train_df, val_df, test_df, metadata).
    """
    import json

    data_dir = Path(data_dir)

    train_df = pd.read_csv(data_dir / f"{ticker}_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(data_dir / f"{ticker}_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv(data_dir / f"{ticker}_test.csv", index_col=0, parse_dates=True)

    with open(data_dir / f"{ticker}_metadata.json", "r") as f:
        metadata = json.load(f)

    return train_df, val_df, test_df, metadata
