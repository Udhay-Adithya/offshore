"""
Tests for metrics computation.

Tests:
- Classification metrics
- Confusion matrix
- ROC-AUC
"""

import pytest
import numpy as np
import torch


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        metrics = mc.compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0

    def test_accuracy_zero(self):
        """Test accuracy with completely wrong predictions."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])

        metrics = mc.compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 computation."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        # Create known scenario
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1])

        metrics = mc.compute_metrics(y_true, y_pred)

        # Check that all metrics are present
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Values should be between 0 and 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_multiclass_metrics(self):
        """Test metrics for multi-class classification."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=5)

        np.random.seed(42)
        y_true = np.random.randint(0, 5, size=100)
        y_pred = np.random.randint(0, 5, size=100)

        metrics = mc.compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])

        cm = mc.confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

        # Check specific values:
        # TN=1, FP=1, FN=1, TP=1
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP

    def test_multiclass_confusion_matrix(self):
        """Test confusion matrix for multi-class."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=5)

        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 0])

        cm = mc.confusion_matrix(y_true, y_pred)

        assert cm.shape == (5, 5)
        assert cm.sum() == len(y_true)

    def test_roc_auc_binary(self):
        """Test ROC-AUC for binary classification."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )

        auc = mc.roc_auc(y_true, y_proba)

        # Perfect ordering should give AUC = 1.0
        assert auc == 1.0

    def test_roc_auc_random(self):
        """Test ROC-AUC with random predictions."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_proba = np.random.rand(100, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        auc = mc.roc_auc(y_true, y_proba)

        # Random predictions should give AUC around 0.5
        assert 0.3 < auc < 0.7

    def test_per_class_metrics(self):
        """Test per-class metrics computation."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=3)

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])

        per_class = mc.per_class_metrics(y_true, y_pred)

        assert len(per_class) == 3

        for class_metrics in per_class.values():
            assert "precision" in class_metrics
            assert "recall" in class_metrics
            assert "f1" in class_metrics


class TestMetricsFromTensors:
    """Test metrics computation from PyTorch tensors."""

    def test_from_tensors(self):
        """Test metrics from tensor inputs."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        y_true = torch.tensor([0, 1, 0, 1, 0, 1])
        y_pred = torch.tensor([0, 1, 0, 1, 1, 0])

        # Convert to numpy
        metrics = mc.compute_metrics(y_true.numpy(), y_pred.numpy())

        assert "accuracy" in metrics
        assert metrics["accuracy"] == pytest.approx(4 / 6, rel=1e-5)

    def test_from_logits(self):
        """Test getting predictions from logits."""
        from src.evaluation.metrics import MetricsCalculator

        mc = MetricsCalculator(num_classes=2)

        # Logits (before softmax)
        logits = torch.tensor(
            [
                [-1.0, 1.0],  # Predicts class 1
                [1.0, -1.0],  # Predicts class 0
                [-2.0, 2.0],  # Predicts class 1
                [2.0, -2.0],  # Predicts class 0
            ]
        )

        y_pred = logits.argmax(dim=1).numpy()
        y_true = np.array([1, 0, 1, 0])

        metrics = mc.compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0


class TestBacktestMetrics:
    """Test backtest-specific metrics."""

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        from src.evaluation.backtest import BacktestEngine

        # Create backtest engine with dummy config
        engine = BacktestEngine(
            initial_capital=100000,
            transaction_cost=0.001,
        )

        # Create returns
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])

        sharpe = engine.calculate_sharpe_ratio(returns, periods_per_year=252)

        # Sharpe should be a number
        assert isinstance(sharpe, float)

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        from src.evaluation.backtest import BacktestEngine

        engine = BacktestEngine(
            initial_capital=100000,
            transaction_cost=0.001,
        )

        # Equity curve: goes up, then down, then up
        equity = np.array([100, 110, 120, 100, 90, 95, 110])

        mdd = engine.calculate_max_drawdown(equity)

        # Max drawdown should be (120-90)/120 = 25%
        assert mdd == pytest.approx(0.25, rel=1e-5)

    def test_win_rate(self):
        """Test win rate calculation."""
        from src.evaluation.backtest import BacktestEngine

        engine = BacktestEngine(
            initial_capital=100000,
            transaction_cost=0.001,
        )

        trade_returns = np.array([0.05, -0.02, 0.03, 0.01, -0.01])

        win_rate = engine.calculate_win_rate(trade_returns)

        # 3 wins out of 5
        assert win_rate == pytest.approx(0.6, rel=1e-5)
