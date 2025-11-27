"""
Tests for feature engineering module.

Tests:
- Technical indicator calculations
- Label creation
"""

import pytest
import numpy as np
import pandas as pd


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""

    def test_compute_returns(self, sample_ohlcv_data):
        """Test returns calculation."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_returns(sample_ohlcv_data)

        assert "returns" in result.columns
        assert "log_returns" in result.columns

        # First value should be NaN (no previous price)
        assert pd.isna(result["returns"].iloc[0])

        # Check calculation
        expected = (result["Close"].iloc[1] - result["Close"].iloc[0]) / result["Close"].iloc[0]
        assert abs(result["returns"].iloc[1] - expected) < 1e-10

    def test_compute_moving_averages(self, sample_ohlcv_data):
        """Test moving average calculations."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_moving_averages(sample_ohlcv_data, windows=[5, 10, 20])

        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "sma_20" in result.columns
        assert "ema_5" in result.columns

        # MA values should be close to price
        assert result["sma_20"].mean() > 0

    def test_compute_rsi(self, sample_ohlcv_data):
        """Test RSI calculation."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_rsi(sample_ohlcv_data, period=14)

        assert "rsi_14" in result.columns

        # RSI should be between 0 and 100
        valid_rsi = result["rsi_14"].dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_compute_macd(self, sample_ohlcv_data):
        """Test MACD calculation."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_macd(sample_ohlcv_data)

        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_compute_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_bollinger_bands(sample_ohlcv_data, window=20, num_std=2)

        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns

        # Upper should be > middle > lower
        valid_idx = result["bb_upper"].notna()
        assert (result.loc[valid_idx, "bb_upper"] >= result.loc[valid_idx, "bb_middle"]).all()
        assert (result.loc[valid_idx, "bb_middle"] >= result.loc[valid_idx, "bb_lower"]).all()

    def test_compute_volatility(self, sample_ohlcv_data):
        """Test volatility calculation."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_volatility(sample_ohlcv_data, window=20)

        assert "volatility_20" in result.columns

        # Volatility should be non-negative
        valid_vol = result["volatility_20"].dropna()
        assert (valid_vol >= 0).all()

    def test_compute_all_features(self, sample_ohlcv_data):
        """Test computing all features at once."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv_data)

        # Should have many features
        assert len(result.columns) > 10

        # Original columns should still exist
        assert "Close" in result.columns
        assert "Volume" in result.columns

    def test_feature_output_shape(self, sample_ohlcv_data):
        """Test that output shape is correct."""
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv_data)

        # Same number of rows
        assert len(result) == len(sample_ohlcv_data)


class TestLabelCreator:
    """Tests for label creation."""

    def test_binary_labels(self, sample_ohlcv_data):
        """Test binary label creation."""
        from src.features.labeling import LabelCreator

        lc = LabelCreator(label_type="binary", horizon=5)
        labels = lc.create_labels(sample_ohlcv_data)

        # Should have labels
        assert len(labels) == len(sample_ohlcv_data)

        # Binary: only 0 and 1 (and NaN at end)
        valid_labels = labels.dropna()
        unique = valid_labels.unique()
        assert set(unique).issubset({0, 1})

    def test_multiclass_labels(self, sample_ohlcv_data):
        """Test multi-class label creation."""
        from src.features.labeling import LabelCreator

        lc = LabelCreator(label_type="multiclass", horizon=5)
        labels = lc.create_labels(sample_ohlcv_data)

        # Should have 5 classes (0-4)
        valid_labels = labels.dropna()
        unique = valid_labels.unique()
        assert all(0 <= u <= 4 for u in unique)

    def test_label_horizon(self, sample_ohlcv_data):
        """Test that labels use correct horizon."""
        from src.features.labeling import LabelCreator

        horizon = 10
        lc = LabelCreator(label_type="binary", horizon=horizon)
        labels = lc.create_labels(sample_ohlcv_data)

        # Last `horizon` labels should be NaN
        assert labels.iloc[-horizon:].isna().all()

    def test_label_alignment_with_features(self, sample_ohlcv_data):
        """Test that labels align with features."""
        from src.features.labeling import LabelCreator
        from src.features.technical import TechnicalIndicators

        ti = TechnicalIndicators()
        features = ti.compute_all(sample_ohlcv_data)

        lc = LabelCreator(label_type="binary", horizon=5)
        labels = lc.create_labels(sample_ohlcv_data)

        # Same index
        assert len(features) == len(labels)
        assert (features.index == labels.index).all()

    def test_custom_thresholds(self, sample_ohlcv_data):
        """Test multi-class with custom thresholds."""
        from src.features.labeling import LabelCreator

        thresholds = [-0.03, -0.01, 0.01, 0.03]  # Custom thresholds
        lc = LabelCreator(label_type="multiclass", horizon=5, thresholds=thresholds)
        labels = lc.create_labels(sample_ohlcv_data)

        valid_labels = labels.dropna()
        assert len(valid_labels) > 0

    def test_label_distribution(self, sample_ohlcv_data):
        """Test that labels have reasonable distribution."""
        from src.features.labeling import LabelCreator

        lc = LabelCreator(label_type="binary", horizon=5)
        labels = lc.create_labels(sample_ohlcv_data)

        valid_labels = labels.dropna()
        value_counts = valid_labels.value_counts(normalize=True)

        # Neither class should be extremely rare (< 5% or > 95%)
        for class_pct in value_counts.values:
            assert 0.05 < class_pct < 0.95
