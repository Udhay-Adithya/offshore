"""Evaluation utilities for Offshore."""

from src.evaluation.metrics import compute_metrics, MetricsCalculator
from src.evaluation.backtest import run_backtest, BacktestEngine
from src.evaluation.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_equity_curve,
)

__all__ = [
    "compute_metrics",
    "MetricsCalculator",
    "run_backtest",
    "BacktestEngine",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_equity_curve",
]
