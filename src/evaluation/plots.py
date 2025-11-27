"""
Plotting utilities for Offshore.

Implements visualization for training curves, confusion matrix, and backtest results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def plot_training_curves(
    history: dict[str, list], save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with training history (loss, accuracy, etc.).
        save_path: Path to save the figure.
        show: Whether to display the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    if "train_accuracy" in history:
        ax.plot(epochs, history["train_accuracy"], label="Train Acc", color="blue")
    if "val_accuracy" in history:
        ax.plot(epochs, history["val_accuracy"], label="Val Acc", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[1, 0]
    if "val_f1" in history:
        ax.plot(epochs, history["val_f1"], label="Val F1", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Validation F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    if "learning_rate" in history:
        ax.plot(epochs, history["learning_rate"], label="LR", color="red")
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        confusion_matrix: Confusion matrix array.
        class_names: List of class names.
        save_path: Path to save the figure.
        show: Whether to display the figure.
        normalize: Whether to normalize the matrix.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed, skipping plot")
        return

    cm = np.array(confusion_matrix)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        fmt = "d"

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_equity_curve(
    equity_curve: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    Plot equity curve from backtest.

    Args:
        equity_curve: Strategy equity curve.
        benchmark: Optional benchmark equity curve (e.g., buy-and-hold).
        timestamps: Optional timestamps for x-axis.
        positions: Optional position array to show on secondary axis.
        save_path: Path to save the figure.
        show: Whether to display the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    x = timestamps if timestamps is not None else np.arange(len(equity_curve))

    # Equity curve
    ax = axes[0]
    ax.plot(x, equity_curve, label="Strategy", color="blue", linewidth=1.5)

    if benchmark is not None:
        ax.plot(x, benchmark, label="Buy & Hold", color="gray", alpha=0.7, linewidth=1)

    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak

    ax = axes[1]
    ax.fill_between(x, drawdown, 0, color="red", alpha=0.3, label="Drawdown")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Equity curve saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    Plot distribution of predictions and probabilities.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        probabilities: Prediction probabilities.
        class_names: List of class names.
        save_path: Path to save the figure.
        show: Whether to display the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    num_classes = probabilities.shape[1] if len(probabilities.shape) > 1 else 2

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # True label distribution
    ax = axes[0]
    unique, counts = np.unique(y_true, return_counts=True)
    ax.bar([class_names[i] for i in unique], counts, color="blue", alpha=0.7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("True Label Distribution")

    # Predicted label distribution
    ax = axes[1]
    unique, counts = np.unique(y_pred, return_counts=True)
    ax.bar([class_names[i] for i in unique], counts, color="orange", alpha=0.7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Label Distribution")

    # Probability distribution
    ax = axes[2]
    if num_classes == 2:
        # Binary: show probability of positive class
        ax.hist(probabilities[:, 1], bins=50, color="green", alpha=0.7)
        ax.axvline(x=0.5, color="red", linestyle="--", label="Threshold 0.5")
        ax.set_xlabel("P(Up)")
    else:
        # Multi-class: show max probability
        max_probs = probabilities.max(axis=1)
        ax.hist(max_probs, bins=50, color="green", alpha=0.7)
        ax.set_xlabel("Max Probability")

    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Prediction distribution saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_returns_over_time(
    timestamps: np.ndarray,
    returns: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    Plot actual returns colored by prediction.

    Args:
        timestamps: Time stamps.
        returns: Actual returns.
        predictions: Model predictions.
        save_path: Path to save the figure.
        show: Whether to display the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color by prediction
    colors = ["red" if p == 0 else "green" for p in predictions]

    ax.scatter(timestamps, returns, c=colors, alpha=0.5, s=10)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Return")
    ax.set_title("Actual Returns Colored by Prediction (Red=Down, Green=Up)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Returns plot saved to {save_path}")

    if show:
        plt.show()

    plt.close()
