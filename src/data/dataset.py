"""
PyTorch Dataset and DataLoader utilities for Offshore.

Handles creating sequence datasets for time-series classification.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock time-series classification.

    Creates sequences of length `lookback` and corresponding labels
    for trend prediction.

    Example:
        >>> dataset = StockDataset(df, feature_cols, lookback=60)
        >>> features, label = dataset[0]
        >>> print(features.shape)  # torch.Size([60, num_features])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        lookback: int = 60,
        label_column: str = "label",
        augmentation: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame with features and labels.
            feature_columns: List of column names to use as features.
            lookback: Number of past time steps to use as input.
            label_column: Name of the label column.
            augmentation: Optional augmentation config with keys:
                         - enabled: bool
                         - noise_std: float (std of Gaussian noise)
        """
        self.lookback = lookback
        self.augmentation = augmentation or {}
        self.feature_columns = feature_columns
        self.label_column = label_column

        # Extract features and labels as numpy arrays
        self.features = df[feature_columns].values.astype(np.float32)
        self.labels = df[label_column].values.astype(np.int64)
        self.timestamps = df.index.values

        # Calculate valid indices (need lookback history)
        self.valid_indices = np.arange(lookback, len(df))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, label):
                - features: Tensor of shape (lookback, num_features)
                - label: Scalar tensor with class label
        """
        # Get the actual index in the data
        data_idx = self.valid_indices[idx]

        # Extract sequence of features
        start_idx = data_idx - self.lookback
        sequence = self.features[start_idx:data_idx]

        # Get label at the end of the sequence
        label = self.labels[data_idx]

        # Convert to tensors
        sequence = torch.from_numpy(sequence)
        label = torch.tensor(label)

        # Apply augmentation (only during training)
        if self.augmentation.get("enabled", False):
            noise_std = self.augmentation.get("noise_std", 0.001)
            noise = torch.randn_like(sequence) * noise_std
            sequence = sequence + noise

        return sequence, label

    def get_timestamp(self, idx: int) -> np.datetime64:
        """Get the timestamp for a sample."""
        data_idx = self.valid_indices[idx]
        return self.timestamps[data_idx]

    @property
    def num_features(self) -> int:
        """Return the number of input features."""
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        """Return the number of unique classes."""
        return len(np.unique(self.labels))

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.

        Uses inverse frequency weighting.

        Returns:
            Tensor of class weights.
        """
        class_counts = np.bincount(self.labels[self.valid_indices])
        total = len(self.valid_indices)
        weights = total / (len(class_counts) * class_counts)
        return torch.from_numpy(weights.astype(np.float32))


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_columns: list[str],
    lookback: int = 60,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentation: Optional[dict[str, Any]] = None,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for training, validation, and test.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame (optional).
        feature_columns: List of feature column names.
        lookback: Lookback window size.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU.
        augmentation: Data augmentation config.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    train_dataset = StockDataset(
        train_df,
        feature_columns,
        lookback=lookback,
        augmentation=augmentation if augmentation else None,
    )

    val_dataset = StockDataset(
        val_df,
        feature_columns,
        lookback=lookback,
        augmentation=None,  # No augmentation for validation
    )

    test_dataset = None
    if test_df is not None:
        test_dataset = StockDataset(
            test_df,
            feature_columns,
            lookback=lookback,
            augmentation=None,  # No augmentation for test
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader


def create_inference_sequences(
    df: pd.DataFrame, feature_columns: list[str], lookback: int = 60
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Create sequences for inference (no labels needed).

    Args:
        df: DataFrame with feature columns.
        feature_columns: List of feature column names.
        lookback: Lookback window size.

    Returns:
        Tuple of (sequences, timestamps):
            - sequences: Tensor of shape (num_samples, lookback, num_features)
            - timestamps: Array of timestamps for each prediction
    """
    features = df[feature_columns].values.astype(np.float32)
    timestamps = df.index.values

    sequences = []
    valid_timestamps = []

    for i in range(lookback, len(df)):
        seq = features[i - lookback : i]
        sequences.append(seq)
        valid_timestamps.append(timestamps[i])

    sequences = torch.from_numpy(np.stack(sequences))
    valid_timestamps = np.array(valid_timestamps)

    return sequences, valid_timestamps
