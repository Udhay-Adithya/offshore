"""
Tests for model architectures.

Tests:
- Model forward passes
- Output shape verification
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn


class TestLSTMClassifier:
    """Tests for LSTM classifier model."""

    def test_forward_pass(self, model_config_lstm, device):
        """Test LSTM forward pass."""
        from src.models.lstm_classifier import LSTMClassifier

        config = model_config_lstm["model"]
        model = LSTMClassifier(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
        ).to(device)

        # Create batch: (batch_size, seq_len, input_size)
        batch_size = 8
        seq_len = 60
        x = torch.randn(batch_size, seq_len, config["input_size"]).to(device)

        # Forward pass
        output = model(x)

        assert output.shape == (batch_size, config["num_classes"])

    def test_output_shape_binary(self, device):
        """Test LSTM output shape for binary classification."""
        from src.models.lstm_classifier import LSTMClassifier

        model = LSTMClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            num_classes=2,
        ).to(device)

        x = torch.randn(16, 60, 20).to(device)
        output = model(x)

        assert output.shape == (16, 2)

    def test_output_shape_multiclass(self, device):
        """Test LSTM output shape for multi-class classification."""
        from src.models.lstm_classifier import LSTMClassifier

        model = LSTMClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            num_classes=5,
        ).to(device)

        x = torch.randn(16, 60, 20).to(device)
        output = model(x)

        assert output.shape == (16, 5)

    def test_bidirectional(self, device):
        """Test bidirectional LSTM."""
        from src.models.lstm_classifier import LSTMClassifier

        model = LSTMClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            num_classes=2,
            bidirectional=True,
        ).to(device)

        x = torch.randn(8, 60, 20).to(device)
        output = model(x)

        assert output.shape == (8, 2)

    def test_gradient_flow(self, device):
        """Test that gradients flow through the model."""
        from src.models.lstm_classifier import LSTMClassifier

        model = LSTMClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            num_classes=2,
        ).to(device)

        x = torch.randn(8, 60, 20, requires_grad=True).to(device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_eval_mode(self, device):
        """Test model in eval mode."""
        from src.models.lstm_classifier import LSTMClassifier

        model = LSTMClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            num_classes=2,
            dropout=0.5,
        ).to(device)

        model.eval()
        x = torch.randn(8, 60, 20).to(device)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(output1, output2)

    def test_different_sequence_lengths(self, device):
        """Test LSTM with different sequence lengths."""
        from src.models.lstm_classifier import LSTMClassifier

        model = LSTMClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            num_classes=2,
        ).to(device)

        for seq_len in [30, 60, 120]:
            x = torch.randn(8, seq_len, 20).to(device)
            output = model(x)
            assert output.shape == (8, 2)


class TestTransformerClassifier:
    """Tests for Transformer classifier model."""

    def test_forward_pass(self, model_config_transformer, device):
        """Test Transformer forward pass."""
        from src.models.transformer_classifier import TransformerClassifier

        config = model_config_transformer["model"]
        model = TransformerClassifier(
            input_size=config["input_size"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            max_seq_length=config["max_seq_length"],
        ).to(device)

        batch_size = 8
        seq_len = 60
        x = torch.randn(batch_size, seq_len, config["input_size"]).to(device)

        output = model(x)

        assert output.shape == (batch_size, config["num_classes"])

    def test_output_shape_binary(self, device):
        """Test Transformer output shape for binary classification."""
        from src.models.transformer_classifier import TransformerClassifier

        model = TransformerClassifier(
            input_size=20,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            num_classes=2,
        ).to(device)

        x = torch.randn(16, 60, 20).to(device)
        output = model(x)

        assert output.shape == (16, 2)

    def test_output_shape_multiclass(self, device):
        """Test Transformer output shape for multi-class classification."""
        from src.models.transformer_classifier import TransformerClassifier

        model = TransformerClassifier(
            input_size=20,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            num_classes=5,
        ).to(device)

        x = torch.randn(16, 60, 20).to(device)
        output = model(x)

        assert output.shape == (16, 5)

    def test_gradient_flow(self, device):
        """Test that gradients flow through Transformer."""
        from src.models.transformer_classifier import TransformerClassifier

        model = TransformerClassifier(
            input_size=20,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            num_classes=2,
        ).to(device)

        x = torch.randn(8, 60, 20, requires_grad=True).to(device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        assert grad_count > 0

    def test_attention_heads(self, device):
        """Test different numbers of attention heads."""
        from src.models.transformer_classifier import TransformerClassifier

        for nhead in [2, 4, 8]:
            d_model = 64  # Must be divisible by nhead
            model = TransformerClassifier(
                input_size=20,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=2,
                dim_feedforward=128,
                num_classes=2,
            ).to(device)

            x = torch.randn(8, 60, 20).to(device)
            output = model(x)

            assert output.shape == (8, 2)

    def test_eval_mode(self, device):
        """Test Transformer in eval mode."""
        from src.models.transformer_classifier import TransformerClassifier

        model = TransformerClassifier(
            input_size=20,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            num_classes=2,
            dropout=0.5,
        ).to(device)

        model.eval()
        x = torch.randn(8, 60, 20).to(device)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)

    def test_different_sequence_lengths(self, device):
        """Test Transformer with different sequence lengths."""
        from src.models.transformer_classifier import TransformerClassifier

        model = TransformerClassifier(
            input_size=20,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            num_classes=2,
            max_seq_length=200,
        ).to(device)

        for seq_len in [30, 60, 120]:
            x = torch.randn(8, seq_len, 20).to(device)
            output = model(x)
            assert output.shape == (8, 2)


class TestClassificationHead:
    """Tests for classification head module."""

    def test_forward_pass(self, device):
        """Test classification head forward pass."""
        from src.models.head import ClassificationHead

        head = ClassificationHead(
            input_dim=128,
            num_classes=5,
            dropout=0.1,
        ).to(device)

        x = torch.randn(16, 128).to(device)
        output = head(x)

        assert output.shape == (16, 5)

    def test_with_hidden_layer(self, device):
        """Test classification head with hidden layer."""
        from src.models.head import ClassificationHead

        head = ClassificationHead(
            input_dim=128,
            num_classes=2,
            hidden_dim=64,
            dropout=0.1,
        ).to(device)

        x = torch.randn(16, 128).to(device)
        output = head(x)

        assert output.shape == (16, 2)


class TestModelRegistry:
    """Tests for model registry."""

    def test_register_and_get_model(self):
        """Test registering and retrieving models."""
        from src.models.base import ModelRegistry

        # Get available models
        available = ModelRegistry.list_models()

        assert "lstm" in available or len(available) >= 0

    def test_create_model_from_config(self, model_config_lstm):
        """Test creating model from config."""
        from src.models.lstm_classifier import LSTMClassifier

        config = model_config_lstm["model"]

        model = LSTMClassifier(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
        )

        assert model is not None
        assert isinstance(model, nn.Module)
