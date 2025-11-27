"""
Training loop for Offshore.

Implements the main training loop with logging, checkpointing, and evaluation.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import BaseClassifier
from src.models.head import get_loss_function
from src.training.callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_config: dict[str, Any]) -> torch.device:
    """Get the appropriate device for training."""
    device_type = device_config.get("type", "auto")

    if device_type == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device_type)


def get_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    optimizer_type = config.get("type", "adamw").lower()
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.01)
    betas = tuple(config.get("betas", [0.9, 0.999]))
    eps = config.get("eps", 1e-8)

    if optimizer_type == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif optimizer_type == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif optimizer_type == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any], num_training_steps: int
) -> Optional[Any]:
    """Create learning rate scheduler from config."""
    scheduler_type = config.get("type", "cosine").lower()

    if scheduler_type == "none":
        return None

    warmup_epochs = config.get("warmup_epochs", 0)
    min_lr = config.get("min_lr", 1e-6)

    if scheduler_type == "cosine":
        # Cosine annealing with warmup
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=num_training_steps - warmup_epochs, eta_min=min_lr
        )

        if warmup_epochs > 0:
            # Linear warmup
            def warmup_lambda(epoch: int) -> float:
                if epoch < warmup_epochs:
                    return float(epoch) / float(max(1, warmup_epochs))
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            return {
                "warmup": warmup_scheduler,
                "main": main_scheduler,
                "warmup_epochs": warmup_epochs,
            }

        return main_scheduler

    elif scheduler_type == "step":
        step_size = config.get("step_size", 30)
        gamma = config.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "plateau":
        patience = config.get("patience", 5)
        factor = config.get("factor", 0.5)
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr  # For F1 score
        )

    return None


class Trainer:
    """
    Main training class for Offshore models.

    Handles the full training loop including:
    - Optimizer and scheduler setup
    - Training and validation epochs
    - Checkpointing and early stopping
    - Metrics logging

    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, config, output_dir)
        >>> history = trainer.train()
    """

    def __init__(
        self,
        model: BaseClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        output_dir: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration.
            output_dir: Directory to save outputs.
            device: Device to train on. If None, auto-detected.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        self.device = device or get_device(config.get("device", {}))
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Training config
        training_config = config.get("training", {})
        self.epochs = training_config.get("epochs", 100)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)

        # Set seed
        seed = config.get("seed", 42)
        set_seed(seed)

        # Optimizer
        optimizer_config = config.get("optimizer", {})
        optimizer_config["learning_rate"] = training_config.get("learning_rate", 0.001)
        self.optimizer = get_optimizer(model, optimizer_config)

        # Scheduler
        scheduler_config = config.get("scheduler", {})
        self.scheduler = get_scheduler(self.optimizer, scheduler_config, self.epochs)

        # Loss function
        loss_config = config.get("loss", {})
        num_classes = model.num_classes

        # Get class weights if configured
        class_weights = None
        if loss_config.get("class_weights") == "auto":
            # Compute from training data
            class_weights = train_loader.dataset.get_class_weights().to(self.device)
            logger.info(f"Using auto class weights: {class_weights}")

        self.criterion = get_loss_function(
            loss_type=loss_config.get("type", "cross_entropy"),
            num_classes=num_classes,
            label_smoothing=loss_config.get("label_smoothing", 0.0),
            class_weights=class_weights,
            focal_gamma=loss_config.get("focal_gamma", 2.0),
        )

        # Callbacks
        self._setup_callbacks(config)

        # Training history
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rate": [],
        }

    def _setup_callbacks(self, config: dict[str, Any]) -> None:
        """Setup training callbacks."""
        # Early stopping
        es_config = config.get("early_stopping", {})
        self.early_stopping = None
        if es_config.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_config.get("patience", 15),
                min_delta=es_config.get("min_delta", 0.001),
                monitor=es_config.get("monitor", "val_f1"),
                mode=es_config.get("mode", "max"),
            )

        # Model checkpoint
        ckpt_config = config.get("checkpoint", {})
        self.checkpoint = ModelCheckpoint(
            save_dir=self.output_dir,
            monitor=ckpt_config.get("monitor", "val_f1"),
            mode=ckpt_config.get("mode", "max"),
            save_best=ckpt_config.get("save_best", True),
            save_last=ckpt_config.get("save_last", True),
        )

        # Logger
        log_config = config.get("logging", {})
        self.training_logger = TrainingLogger(
            log_interval=log_config.get("log_interval", 10), output_dir=self.output_dir
        )

    def train(self) -> dict[str, list]:
        """
        Run the full training loop.

        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")

        best_val_metric = float("-inf")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Training epoch
            train_loss, train_acc = self._train_epoch(epoch)

            # Validation epoch
            val_loss, val_acc, val_f1, val_metrics = self._validate_epoch()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["learning_rate"].append(current_lr)

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}, LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.1f}s"
            )

            # Update scheduler
            self._update_scheduler(epoch, val_f1)

            # Checkpoint
            metrics = {"val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1}
            self.checkpoint.step(self.model, self.optimizer, epoch, metrics)

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping.step(
                    val_f1 if self.early_stopping.mode == "max" else val_loss
                ):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Save final results
        self._save_results()

        return self.history

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        self.optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self) -> tuple[float, float, float, dict[str, Any]]:
        """Run one validation epoch."""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probs = []

        for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=-1)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        # Compute metrics
        metrics = compute_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probs),
            num_classes=self.model.num_classes,
        )

        return avg_loss, metrics["accuracy"], metrics["f1_macro"], metrics

    def _update_scheduler(self, epoch: int, val_metric: float) -> None:
        """Update learning rate scheduler."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, dict):
            # Warmup + main scheduler
            if epoch <= self.scheduler["warmup_epochs"]:
                self.scheduler["warmup"].step()
            else:
                self.scheduler["main"].step()
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_metric)
        else:
            self.scheduler.step()

    def _save_results(self) -> None:
        """Save training results and history."""
        # Save history
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        # Plot training curves if matplotlib available
        try:
            from src.evaluation.plots import plot_training_curves

            plot_training_curves(self.history, save_path=self.output_dir / "training_curves.png")
        except ImportError:
            logger.warning("Could not plot training curves (matplotlib not available)")

        logger.info(f"Training results saved to {self.output_dir}")


def train_model(
    model: BaseClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    output_dir: Union[str, Path],
) -> dict[str, list]:
    """
    Train a model with the given configuration.

    Convenience function that creates a Trainer and runs training.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        output_dir: Directory to save outputs.

    Returns:
        Training history dictionary.
    """
    trainer = Trainer(model, train_loader, val_loader, config, output_dir)
    return trainer.train()
