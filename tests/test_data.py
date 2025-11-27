"""
Tests for data module.

Tests:
- Dataset creation and slicing
- Label alignment
- Data loader creation
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader


class TestStockDataset:
    """Tests for StockDataset class."""

    def test_dataset_creation(self, sample_features, sample_labels_binary):
        """Test basic dataset creation."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)

        assert len(dataset) == len(sample_labels_binary)
        assert len(dataset) == sample_features.shape[0]

    def test_dataset_getitem(self, sample_features, sample_labels_binary):
        """Test getting items from dataset."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (60, 20)  # sequence_length, n_features
        assert y.shape == ()  # scalar

    def test_dataset_slicing(self, sample_features, sample_labels_binary):
        """Test that slicing returns correct indices."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)

        # Test multiple indices
        indices = [0, 10, 50, 99]
        for idx in indices:
            x, y = dataset[idx]

            # Verify feature alignment
            expected_x = torch.from_numpy(sample_features[idx])
            assert torch.allclose(x, expected_x)

            # Verify label alignment
            expected_y = torch.tensor(sample_labels_binary[idx])
            assert y == expected_y

    def test_dataset_label_alignment(self, sample_features, sample_labels_multiclass):
        """Test that labels are correctly aligned with features."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_multiclass)

        # Labels should be in valid range
        for i in range(len(dataset)):
            _, y = dataset[i]
            assert 0 <= y.item() < 5

    def test_dataset_dtype(self, sample_features, sample_labels_binary):
        """Test correct data types."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)

        x, y = dataset[0]

        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_dataset_with_transforms(self, sample_features, sample_labels_binary):
        """Test dataset with optional transforms."""
        from src.data.dataset import StockDataset

        def normalize_transform(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        dataset = StockDataset(sample_features, sample_labels_binary, transform=normalize_transform)

        x, _ = dataset[0]

        # Check that transform was applied (mean should be ~0, std ~1)
        assert abs(x.mean().item()) < 0.5

    def test_dataset_empty(self):
        """Test handling of empty dataset."""
        from src.data.dataset import StockDataset

        empty_features = np.array([]).reshape(0, 60, 20).astype(np.float32)
        empty_labels = np.array([]).astype(np.int64)

        dataset = StockDataset(empty_features, empty_labels)

        assert len(dataset) == 0


class TestDataLoaders:
    """Tests for DataLoader creation."""

    def test_create_dataloaders(self, sample_features, sample_labels_binary):
        """Test creating train/val/test dataloaders."""
        from src.data.dataset import StockDataset, create_dataloaders

        # Split data
        n_samples = len(sample_labels_binary)
        train_idx = int(n_samples * 0.7)
        val_idx = int(n_samples * 0.85)

        train_features = sample_features[:train_idx]
        train_labels = sample_labels_binary[:train_idx]
        val_features = sample_features[train_idx:val_idx]
        val_labels = sample_labels_binary[train_idx:val_idx]
        test_features = sample_features[val_idx:]
        test_labels = sample_labels_binary[val_idx:]

        train_loader, val_loader, test_loader = create_dataloaders(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            test_features=test_features,
            test_labels=test_labels,
            batch_size=16,
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_dataloader_batch_shape(self, sample_features, sample_labels_binary):
        """Test that dataloader returns correct batch shapes."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        batch_x, batch_y = next(iter(loader))

        assert batch_x.shape == (16, 60, 20)  # batch, seq_len, features
        assert batch_y.shape == (16,)

    def test_dataloader_iteration(self, sample_features, sample_labels_binary):
        """Test iterating through all batches."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        total_samples = 0
        for batch_x, batch_y in loader:
            total_samples += batch_x.shape[0]

        assert total_samples == len(dataset)


class TestLabelAlignment:
    """Tests specifically for label alignment issues."""

    def test_label_feature_count_match(self, sample_features, sample_labels_binary):
        """Test that feature and label counts match."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)

        # Should have same number of features and labels
        assert sample_features.shape[0] == len(sample_labels_binary)
        assert len(dataset) == len(sample_labels_binary)

    def test_label_indices_preserved(self, sample_features, sample_labels_binary):
        """Test that label indices are preserved correctly."""
        from src.data.dataset import StockDataset

        dataset = StockDataset(sample_features, sample_labels_binary)

        # Create known label pattern
        known_labels = np.array([0, 1, 0, 1, 0] * 20, dtype=np.int64)
        known_features = np.random.randn(100, 60, 20).astype(np.float32)

        dataset_known = StockDataset(known_features, known_labels)

        # Verify pattern is preserved
        for i in range(len(known_labels)):
            _, y = dataset_known[i]
            assert y.item() == known_labels[i]

    def test_multiclass_label_distribution(self, sample_labels_multiclass):
        """Test that all classes are represented."""
        unique_labels = np.unique(sample_labels_multiclass)

        # Should have labels from 0 to 4
        assert len(unique_labels) >= 3  # At least 3 classes should appear with seed=42
        assert max(unique_labels) <= 4
        assert min(unique_labels) >= 0
