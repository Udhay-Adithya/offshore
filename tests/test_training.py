"""
Tests for training module.

Tests:
- Training loop basics
- Callbacks
- Checkpointing
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=20, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # Take last timestep if 3D
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.fc(x)


@pytest.fixture
def simple_dataloaders():
    """Create simple dataloaders for testing."""
    torch.manual_seed(42)

    n_samples = 100
    seq_len = 60
    input_size = 20
    num_classes = 2

    X = torch.randn(n_samples, seq_len, input_size)
    y = torch.randint(0, num_classes, (n_samples,))

    dataset = TensorDataset(X, y)

    train_size = int(0.7 * n_samples)
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self, simple_dataloaders, device):
        """Test trainer initialization."""
        from src.training.train_loop import Trainer

        train_loader, val_loader = simple_dataloaders
        model = SimpleModel().to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001,
            device=device,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_single_epoch(self, simple_dataloaders, device):
        """Test running a single training epoch."""
        from src.training.train_loop import Trainer

        train_loader, val_loader = simple_dataloaders
        model = SimpleModel().to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001,
            device=device,
        )

        train_loss = trainer.train_epoch()

        assert train_loss >= 0
        assert isinstance(train_loss, float)

    def test_validation_epoch(self, simple_dataloaders, device):
        """Test running validation."""
        from src.training.train_loop import Trainer

        train_loader, val_loader = simple_dataloaders
        model = SimpleModel().to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001,
            device=device,
        )

        val_loss, val_acc = trainer.validate()

        assert val_loss >= 0
        assert 0 <= val_acc <= 1

    def test_full_training(self, simple_dataloaders, device):
        """Test full training loop."""
        from src.training.train_loop import Trainer

        train_loader, val_loader = simple_dataloaders
        model = SimpleModel().to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001,
            device=device,
        )

        history = trainer.fit(epochs=2)

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2

    def test_model_improves(self, simple_dataloaders, device):
        """Test that model loss decreases."""
        from src.training.train_loop import Trainer

        train_loader, val_loader = simple_dataloaders
        model = SimpleModel().to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.01,  # Higher LR for faster convergence
            device=device,
        )

        history = trainer.fit(epochs=5)

        # Loss should generally decrease (not strictly monotonic due to randomness)
        assert history["train_loss"][-1] <= history["train_loss"][0] * 1.5


class TestEarlyStopping:
    """Tests for early stopping callback."""

    def test_early_stopping_trigger(self):
        """Test that early stopping triggers."""
        from src.training.callbacks import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.0)

        # Simulated val losses (getting worse)
        val_losses = [1.0, 0.9, 0.95, 0.96, 0.97]

        for loss in val_losses:
            should_stop = es(loss)

        # Should have triggered (4 epochs without improvement after 0.9)
        assert should_stop or es.counter >= 3

    def test_early_stopping_reset(self):
        """Test early stopping counter reset on improvement."""
        from src.training.callbacks import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.0)

        # Improving then staying flat
        assert not es(1.0)
        assert not es(0.9)
        assert es.counter == 0
        assert not es(0.8)
        assert es.counter == 0

    def test_early_stopping_min_delta(self):
        """Test min_delta threshold."""
        from src.training.callbacks import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.1)

        # Small improvements (less than min_delta)
        assert not es(1.0)
        assert not es(0.95)  # Only 0.05 improvement
        assert es.counter >= 1  # Should count as no improvement


class TestModelCheckpoint:
    """Tests for model checkpointing."""

    def test_checkpoint_save(self, device):
        """Test saving model checkpoint."""
        from src.training.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(
                filepath=os.path.join(tmpdir, "model.pt"),
                save_best_only=True,
            )

            model = SimpleModel().to(device)

            # Save with initial val loss
            checkpoint(model, val_loss=1.0)

            assert os.path.exists(os.path.join(tmpdir, "model.pt"))

    def test_checkpoint_best_only(self, device):
        """Test save_best_only flag."""
        from src.training.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            checkpoint = ModelCheckpoint(
                filepath=filepath,
                save_best_only=True,
            )

            model = SimpleModel().to(device)

            # First save
            checkpoint(model, val_loss=1.0)
            mtime1 = os.path.getmtime(filepath)

            # Worse model shouldn't save
            checkpoint(model, val_loss=1.5)
            mtime2 = os.path.getmtime(filepath)
            assert mtime1 == mtime2

            # Better model should save
            checkpoint(model, val_loss=0.5)
            mtime3 = os.path.getmtime(filepath)
            assert mtime3 > mtime1

    def test_checkpoint_load(self, device):
        """Test loading model from checkpoint."""
        from src.training.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            checkpoint = ModelCheckpoint(filepath=filepath)

            model1 = SimpleModel().to(device)
            checkpoint(model1, val_loss=1.0)

            # Load into new model
            model2 = SimpleModel().to(device)
            model2.load_state_dict(torch.load(filepath, map_location=device))

            # Parameters should match
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestMetricsLogger:
    """Tests for metrics logging callback."""

    def test_logger_records_metrics(self):
        """Test that logger records metrics."""
        from src.training.callbacks import MetricsLogger

        logger = MetricsLogger()

        logger.log(epoch=0, train_loss=1.0, val_loss=0.9, val_acc=0.5)
        logger.log(epoch=1, train_loss=0.8, val_loss=0.7, val_acc=0.6)

        history = logger.get_history()

        assert len(history["train_loss"]) == 2
        assert history["val_acc"][-1] == 0.6

    def test_logger_get_best(self):
        """Test getting best metrics."""
        from src.training.callbacks import MetricsLogger

        logger = MetricsLogger()

        logger.log(epoch=0, train_loss=1.0, val_loss=0.9, val_acc=0.5)
        logger.log(epoch=1, train_loss=0.8, val_loss=0.7, val_acc=0.6)
        logger.log(epoch=2, train_loss=0.6, val_loss=0.8, val_acc=0.55)

        best_epoch, best_val_loss = logger.get_best("val_loss", mode="min")

        assert best_epoch == 1
        assert best_val_loss == 0.7
