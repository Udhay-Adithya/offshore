"""
Pytest configuration and fixtures.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Generate realistic-ish price data
    initial_price = 100.0
    returns = np.random.randn(n_days) * 0.02  # 2% daily volatility
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    high = prices * (1 + np.abs(np.random.randn(n_days) * 0.01))
    low = prices * (1 - np.abs(np.random.randn(n_days) * 0.01))
    open_prices = low + (high - low) * np.random.rand(n_days)
    close_prices = low + (high - low) * np.random.rand(n_days)
    volume = np.random.randint(1000000, 10000000, size=n_days)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close_prices,
            "Adj Close": close_prices,
            "Volume": volume,
        }
    )
    df.set_index("Date", inplace=True)

    return df


@pytest.fixture
def sample_features() -> np.ndarray:
    """Create sample feature array for testing."""
    np.random.seed(42)
    # (samples, sequence_length, n_features)
    return np.random.randn(100, 60, 20).astype(np.float32)


@pytest.fixture
def sample_labels_binary() -> np.ndarray:
    """Create sample binary labels."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=100).astype(np.int64)


@pytest.fixture
def sample_labels_multiclass() -> np.ndarray:
    """Create sample multi-class labels (5 classes)."""
    np.random.seed(42)
    return np.random.randint(0, 5, size=100).astype(np.int64)


@pytest.fixture
def model_config_lstm() -> dict:
    """LSTM model configuration for testing."""
    return {
        "model": {
            "type": "lstm",
            "input_size": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": False,
            "num_classes": 2,
        }
    }


@pytest.fixture
def model_config_transformer() -> dict:
    """Transformer model configuration for testing."""
    return {
        "model": {
            "type": "transformer",
            "input_size": 20,
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "num_classes": 2,
            "max_seq_length": 100,
        }
    }


@pytest.fixture
def train_config() -> dict:
    """Training configuration for testing."""
    return {
        "training": {
            "seed": 42,
            "batch_size": 16,
            "epochs": 2,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "weight_decay": 0.01,
            "scheduler": None,
            "early_stopping_patience": 5,
            "label_smoothing": 0.0,
            "gradient_clip_norm": 1.0,
        }
    }


@pytest.fixture
def data_config() -> dict:
    """Data configuration for testing."""
    return {
        "data": {
            "ticker": "TEST",
            "interval": "1d",
            "lookback_window": 60,
            "prediction_horizon": 5,
            "label_type": "binary",
            "train_end_date": "2020-10-01",
            "val_end_date": "2020-12-01",
        }
    }


@pytest.fixture
def device() -> torch.device:
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
