"""
Training callbacks for Offshore.

Implements early stopping, checkpointing, and logging callbacks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to stop training when metric stops improving.

    Example:
        >>> early_stop = EarlyStopping(patience=10, monitor="val_f1", mode="max")
        >>> for epoch in range(epochs):
        ...     val_f1 = train_epoch()
        ...     if early_stop.step(val_f1):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            monitor: Metric name to monitor.
            mode: "min" if lower is better, "max" if higher is better.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def step(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.early_stop = False


class ModelCheckpoint:
    """
    Model checkpoint callback to save model during training.

    Saves the best model based on a monitored metric and optionally
    the last model at each epoch.

    Example:
        >>> checkpoint = ModelCheckpoint(save_dir="outputs", monitor="val_f1")
        >>> for epoch in range(epochs):
        ...     metrics = {"val_f1": val_f1, "val_loss": val_loss}
        ...     checkpoint.step(model, optimizer, epoch, metrics)
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
        filename_prefix: str = "",
    ):
        """
        Initialize model checkpoint.

        Args:
            save_dir: Directory to save checkpoints.
            monitor: Metric name to monitor.
            mode: "min" if lower is better, "max" if higher is better.
            save_best: Whether to save the best model.
            save_last: Whether to save the last model each epoch.
            filename_prefix: Prefix for checkpoint filenames.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.filename_prefix = filename_prefix

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> bool:
        """
        Save checkpoint if necessary.

        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch.
            metrics: Dictionary of metric values.

        Returns:
            True if a new best model was saved, False otherwise.
        """
        saved_best = False
        current_value = metrics.get(self.monitor, 0)

        # Check if this is the best model
        if self.mode == "min":
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value

        # Save best model
        if self.save_best and is_best:
            self.best_value = current_value
            self.best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "best_value": self.best_value,
            }

            # Add model config if available
            if hasattr(model, "input_dim"):
                checkpoint["model_config"] = {
                    "input_dim": model.input_dim,
                    "seq_length": model.seq_length,
                    "num_classes": model.num_classes,
                }

            prefix = f"{self.filename_prefix}_" if self.filename_prefix else ""
            best_path = self.save_dir / f"{prefix}best_model.pt"
            torch.save(checkpoint, best_path)

            logger.info(
                f"New best model saved: {self.monitor}={current_value:.4f} " f"(epoch {epoch})"
            )
            saved_best = True

        # Save last model
        if self.save_last:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }

            if hasattr(model, "input_dim"):
                checkpoint["model_config"] = {
                    "input_dim": model.input_dim,
                    "seq_length": model.seq_length,
                    "num_classes": model.num_classes,
                }

            prefix = f"{self.filename_prefix}_" if self.filename_prefix else ""
            last_path = self.save_dir / f"{prefix}last_model.pt"
            torch.save(checkpoint, last_path)

        return saved_best

    def load_best(
        self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> dict[str, Any]:
        """
        Load the best checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to load state into (optional).

        Returns:
            Checkpoint dictionary with metadata.
        """
        prefix = f"{self.filename_prefix}_" if self.filename_prefix else ""
        best_path = self.save_dir / f"{prefix}best_model.pt"

        if not best_path.exists():
            raise FileNotFoundError(f"Best model not found at {best_path}")

        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint


class LearningRateScheduler:
    """
    Learning rate scheduler callback with warmup support.

    Wraps PyTorch schedulers with additional warmup functionality.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        **kwargs: Any,
    ):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule.
            scheduler_type: Type of scheduler ("cosine", "step", "plateau").
            warmup_epochs: Number of warmup epochs.
            total_epochs: Total number of training epochs.
            min_lr: Minimum learning rate.
            **kwargs: Additional scheduler-specific arguments.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

        # Store initial learning rate
        self.base_lr = optimizer.param_groups[0]["lr"]

        # Create scheduler
        self.scheduler = self._create_scheduler(scheduler_type, **kwargs)

    def _create_scheduler(self, scheduler_type: str, **kwargs: Any) -> Optional[Any]:
        """Create the appropriate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=self.total_epochs - self.warmup_epochs, eta_min=self.min_lr
            )
        elif scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=kwargs.get("mode", "max"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5),
                min_lr=self.min_lr,
            )
        return None

    def step(self, epoch: int, metric: Optional[float] = None) -> float:
        """
        Update learning rate.

        Args:
            epoch: Current epoch (1-indexed).
            metric: Metric value for plateau scheduler.

        Returns:
            Current learning rate.
        """
        # Warmup phase
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.base_lr * warmup_factor

        # Main scheduler
        elif self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()

        return self.optimizer.param_groups[0]["lr"]


class TrainingLogger:
    """
    Training logger callback for detailed logging.

    Logs metrics at specified intervals and saves logs to file.
    """

    def __init__(self, log_interval: int = 10, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize training logger.

        Args:
            log_interval: Log every N batches.
            output_dir: Directory to save log files.
        """
        self.log_interval = log_interval
        self.output_dir = Path(output_dir) if output_dir else None

        self.batch_logs: list[dict[str, Any]] = []
        self.epoch_logs: list[dict[str, Any]] = []

    def log_batch(
        self, epoch: int, batch: int, loss: float, accuracy: float, learning_rate: float
    ) -> None:
        """Log batch-level metrics."""
        if batch % self.log_interval == 0:
            log_entry = {
                "epoch": epoch,
                "batch": batch,
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": learning_rate,
            }
            self.batch_logs.append(log_entry)

            logger.debug(
                f"Epoch {epoch} Batch {batch}: "
                f"Loss={loss:.4f}, Acc={accuracy:.4f}, LR={learning_rate:.6f}"
            )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        val_f1: float,
        learning_rate: float,
        **extra_metrics: float,
    ) -> None:
        """Log epoch-level metrics."""
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "learning_rate": learning_rate,
            **extra_metrics,
        }
        self.epoch_logs.append(log_entry)

    def save_logs(self) -> None:
        """Save logs to file."""
        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save epoch logs
        epoch_log_path = self.output_dir / "epoch_logs.json"
        with open(epoch_log_path, "w") as f:
            json.dump(self.epoch_logs, f, indent=2)

        # Save batch logs if not too large
        if len(self.batch_logs) < 10000:
            batch_log_path = self.output_dir / "batch_logs.json"
            with open(batch_log_path, "w") as f:
                json.dump(self.batch_logs, f, indent=2)
