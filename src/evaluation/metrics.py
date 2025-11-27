"""
Classification metrics for Offshore.

Implements accuracy, precision, recall, F1, confusion matrix, and ROC-AUC.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


class MetricsCalculator:
    """
    Calculate classification metrics for model evaluation.

    Example:
        >>> calculator = MetricsCalculator(num_classes=2)
        >>> metrics = calculator.compute(y_true, y_pred, y_probs)
    """

    def __init__(self, num_classes: int = 2, average: str = "macro"):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes.
            average: Averaging method for multi-class ("macro", "micro", "weighted").
        """
        self.num_classes = num_classes
        self.average = average
        self.class_names = self._get_class_names()

    def _get_class_names(self) -> list[str]:
        """Get class names based on number of classes."""
        if self.num_classes == 2:
            return ["down", "up"]
        elif self.num_classes == 3:
            return ["down", "flat", "up"]
        elif self.num_classes == 5:
            return ["strong_down", "mild_down", "flat", "mild_up", "strong_up"]
        else:
            return [f"class_{i}" for i in range(self.num_classes)]

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None
    ) -> dict[str, Any]:
        """
        Compute all classification metrics.

        Args:
            y_true: True labels of shape (n_samples,).
            y_pred: Predicted labels of shape (n_samples,).
            y_probs: Predicted probabilities of shape (n_samples, num_classes).
                    Optional, required for ROC-AUC.

        Returns:
            Dictionary of metrics.
        """
        metrics: dict[str, Any] = {}

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        metrics["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics["per_class"] = {}
        for i, name in enumerate(self.class_names[: len(per_class_precision)]):
            metrics["per_class"][name] = {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names[: cm.shape[0]],
            zero_division=0,
            output_dict=True,
        )

        # ROC-AUC (if probabilities provided)
        if y_probs is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_probs[:, 1]))
                else:
                    # Multi-class
                    metrics["roc_auc_ovr"] = float(
                        roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
                    )
            except ValueError as e:
                # ROC-AUC may fail if not all classes present
                metrics["roc_auc_error"] = str(e)

        # Additional statistics
        metrics["num_samples"] = len(y_true)
        metrics["class_distribution"] = {
            name: int((y_true == i).sum())
            for i, name in enumerate(self.class_names[: self.num_classes])
        }

        return metrics

    def print_summary(self, metrics: dict[str, Any]) -> None:
        """Print a human-readable summary of metrics."""
        print("\n" + "=" * 60)
        print("Classification Metrics Summary")
        print("=" * 60)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1 Score (macro):   {metrics['f1_macro']:.4f}")

        if "roc_auc" in metrics:
            print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        elif "roc_auc_ovr" in metrics:
            print(f"  ROC-AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")

        print(f"\nPer-Class Metrics:")
        for class_name, class_metrics in metrics.get("per_class", {}).items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1:        {class_metrics['f1']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        print(f"  {cm}")

        print("\n" + "=" * 60)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 2,
    average: str = "macro",
) -> dict[str, Any]:
    """
    Compute classification metrics.

    Convenience function that creates a MetricsCalculator and computes metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_probs: Predicted probabilities (optional).
        num_classes: Number of classes.
        average: Averaging method.

    Returns:
        Dictionary of metrics.
    """
    calculator = MetricsCalculator(num_classes=num_classes, average=average)
    return calculator.compute(y_true, y_pred, y_probs)


def compute_directional_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2
) -> dict[str, float]:
    """
    Compute directional accuracy metrics for trading.

    For binary: correct direction (up/down)
    For multi-class: correct trend direction (up/flat/down collapsed)

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.

    Returns:
        Dictionary with directional metrics.
    """
    if num_classes == 2:
        # Binary: 0=down, 1=up
        correct_direction = (y_true == y_pred).mean()
        up_accuracy = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).sum() > 0 else 0
        down_accuracy = (y_pred[y_true == 0] == 0).mean() if (y_true == 0).sum() > 0 else 0

        return {
            "directional_accuracy": float(correct_direction),
            "up_accuracy": float(up_accuracy),
            "down_accuracy": float(down_accuracy),
        }

    elif num_classes == 5:
        # Multi-class: collapse to direction
        # 0,1 = down, 2 = flat, 3,4 = up
        true_direction = np.where(y_true < 2, 0, np.where(y_true > 2, 2, 1))
        pred_direction = np.where(y_pred < 2, 0, np.where(y_pred > 2, 2, 1))

        correct_direction = (true_direction == pred_direction).mean()

        return {
            "directional_accuracy": float(correct_direction),
            "collapsed_accuracy": float((true_direction == pred_direction).mean()),
        }

    return {"directional_accuracy": float((y_true == y_pred).mean())}
