"""
Label creation for trend classification in Offshore.

Implements binary and multi-class labeling based on future returns.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd


class LabelCreator:
    """
    Create labels for stock trend classification.

    Supports:
    - Binary: up (1) / down (0)
    - Multi-class: strong_down, mild_down, flat, mild_up, strong_up

    Example:
        >>> creator = LabelCreator(horizon=5, label_type="binary")
        >>> df = creator.create(df)
    """

    def __init__(
        self,
        horizon: int = 5,
        label_type: str = "binary",
        binary_threshold: float = 0.0,
        multiclass_config: Optional[dict[str, Any]] = None,
        price_col: str = "close",
    ):
        """
        Initialize the label creator.

        Args:
            horizon: Number of periods ahead to calculate returns.
            label_type: "binary" or "multiclass".
            binary_threshold: Return threshold for binary classification.
                            If return > threshold: up (1)
                            If return <= threshold: down (0)
            multiclass_config: Config for multi-class labeling with keys:
                - num_classes: Number of classes (default 5)
                - threshold_type: "percentile" or "fixed"
                - percentile_thresholds: List of percentiles [10, 30, 70, 90]
                - fixed_thresholds: List of return thresholds [-0.02, -0.005, 0.005, 0.02]
            price_col: Column to use for return calculation.
        """
        self.horizon = horizon
        self.label_type = label_type
        self.binary_threshold = binary_threshold
        self.multiclass_config = multiclass_config or {}
        self.price_col = price_col

        # Validate
        if label_type not in ["binary", "multiclass"]:
            raise ValueError(f"Invalid label_type: {label_type}")

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for the DataFrame.

        Args:
            df: DataFrame with price data.

        Returns:
            DataFrame with added 'label' and 'future_return' columns.
        """
        df = df.copy()

        # Calculate future returns
        df["future_return"] = df[self.price_col].shift(-self.horizon) / df[self.price_col] - 1

        # Create labels based on type
        if self.label_type == "binary":
            df["label"] = self._create_binary_labels(df["future_return"])
        else:
            df["label"] = self._create_multiclass_labels(df["future_return"])

        return df

    def _create_binary_labels(self, returns: pd.Series) -> pd.Series:
        """
        Create binary labels (0: down, 1: up).

        Args:
            returns: Series of future returns.

        Returns:
            Series of binary labels.
        """
        return (returns > self.binary_threshold).astype(int)

    def _create_multiclass_labels(self, returns: pd.Series) -> pd.Series:
        """
        Create multi-class labels based on return buckets.

        Classes:
            0: strong_down
            1: mild_down
            2: flat
            3: mild_up
            4: strong_up
        """
        num_classes = self.multiclass_config.get("num_classes", 5)
        threshold_type = self.multiclass_config.get("threshold_type", "percentile")

        if threshold_type == "percentile":
            # Use percentile-based thresholds
            percentiles = self.multiclass_config.get("percentile_thresholds", [10, 30, 70, 90])
            thresholds = np.nanpercentile(returns.dropna(), percentiles)
        else:
            # Use fixed thresholds
            thresholds = self.multiclass_config.get(
                "fixed_thresholds", [-0.02, -0.005, 0.005, 0.02]
            )

        # Create labels using digitize
        # digitize returns indices where values should be inserted
        # to maintain sorted order
        labels = np.digitize(returns, thresholds)

        return pd.Series(labels, index=returns.index)

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        if self.label_type == "binary":
            return 2
        return self.multiclass_config.get("num_classes", 5)

    @staticmethod
    def get_class_names(label_type: str, num_classes: int = 5) -> list[str]:
        """
        Get human-readable class names.

        Args:
            label_type: "binary" or "multiclass".
            num_classes: Number of classes for multiclass.

        Returns:
            List of class names.
        """
        if label_type == "binary":
            return ["down", "up"]
        elif num_classes == 3:
            return ["down", "flat", "up"]
        elif num_classes == 5:
            return ["strong_down", "mild_down", "flat", "mild_up", "strong_up"]
        else:
            return [f"class_{i}" for i in range(num_classes)]


def create_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    label_type: str = "binary",
    binary_threshold: float = 0.0,
    multiclass_config: Optional[dict[str, Any]] = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Create labels for trend classification.

    Convenience function that creates a LabelCreator and adds labels.

    Args:
        df: DataFrame with price data.
        horizon: Prediction horizon in periods.
        label_type: "binary" or "multiclass".
        binary_threshold: Threshold for binary classification.
        multiclass_config: Config for multi-class labeling.
        price_col: Price column to use.

    Returns:
        DataFrame with 'label' and 'future_return' columns added.

    Example:
        >>> df = create_labels(df, horizon=5, label_type="binary")
        >>> print(df["label"].value_counts())
    """
    creator = LabelCreator(
        horizon=horizon,
        label_type=label_type,
        binary_threshold=binary_threshold,
        multiclass_config=multiclass_config,
        price_col=price_col,
    )
    return creator.create(df)


def get_label_distribution(df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    """
    Get label distribution statistics.

    Args:
        df: DataFrame with labels.
        label_col: Name of label column.

    Returns:
        Dictionary with distribution statistics.
    """
    labels = df[label_col].dropna()

    value_counts = labels.value_counts().sort_index()
    percentages = (value_counts / len(labels) * 100).round(2)

    return {
        "counts": value_counts.to_dict(),
        "percentages": percentages.to_dict(),
        "total": len(labels),
        "num_classes": len(value_counts),
        "most_common": value_counts.idxmax(),
        "least_common": value_counts.idxmin(),
        "imbalance_ratio": value_counts.max() / value_counts.min(),
    }
