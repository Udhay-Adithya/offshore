"""
Model inference utilities for Offshore.

Handles loading trained models and making predictions on new data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.models import ModelRegistry
from src.models.base import BaseClassifier


class Predictor:
    """
    Predictor class for making inference with trained models.

    Example:
        >>> predictor = Predictor.from_checkpoint("path/to/model.pt")
        >>> probs, pred = predictor.predict(sequence)
    """

    def __init__(
        self,
        model: BaseClassifier,
        feature_columns: list[str],
        scaler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model: Trained model.
            feature_columns: List of feature column names.
            scaler: Optional fitted scaler for feature normalization.
            device: Device to run inference on.
        """
        self.model = model
        self.feature_columns = feature_columns
        self.scaler = scaler

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model_type: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> "Predictor":
        """
        Load a predictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint.
            model_type: Model type ("transformer", "lstm"). Auto-detected if None.
            feature_columns: Feature columns. If None, tries to load from metadata.
            scaler_path: Path to fitted scaler. If None, no scaling applied.
            device: Device to run on.

        Returns:
            Predictor instance.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint.get("model_config", {})

        # Auto-detect model type
        if model_type is None:
            path_str = str(checkpoint_path).lower()
            if "transformer" in path_str:
                model_type = "transformer"
            elif "lstm" in path_str or "gru" in path_str:
                model_type = "lstm"
            else:
                model_type = "transformer"  # Default

        # Create model
        model = ModelRegistry.create(
            model_type,
            input_dim=model_config.get("input_dim", 32),
            seq_length=model_config.get("seq_length", 60),
            num_classes=model_config.get("num_classes", 2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load feature columns from metadata if not provided
        if feature_columns is None:
            metadata_path = Path(checkpoint_path).parent.parent / "metadata.json"
            if metadata_path.exists():
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
                feature_columns = metadata.get("feature_columns", [])
            else:
                feature_columns = []

        # Load scaler if provided
        scaler = None
        if scaler_path is not None:
            import joblib

            scaler = joblib.load(scaler_path)

        return cls(model, feature_columns, scaler, device)

    def predict(self, sequence: Union[np.ndarray, torch.Tensor]) -> tuple[np.ndarray, int]:
        """
        Make prediction on a single sequence.

        Args:
            sequence: Input sequence of shape (seq_length, num_features)
                     or (1, seq_length, num_features).

        Returns:
            Tuple of (probabilities, predicted_class).
        """
        # Ensure tensor format
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence.astype(np.float32))

        # Add batch dimension if needed
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)

        sequence = sequence.to(self.device)

        with torch.no_grad():
            output = self.model(sequence)
            probs = torch.softmax(output, dim=-1)
            pred = output.argmax(dim=-1)

        return probs.cpu().numpy()[0], pred.cpu().item()

    def predict_batch(
        self, sequences: Union[np.ndarray, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of sequences.

        Args:
            sequences: Input sequences of shape (batch, seq_length, num_features).

        Returns:
            Tuple of (probabilities, predicted_classes).
        """
        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(sequences.astype(np.float32))

        sequences = sequences.to(self.device)

        with torch.no_grad():
            output = self.model(sequences)
            probs = torch.softmax(output, dim=-1)
            preds = output.argmax(dim=-1)

        return probs.cpu().numpy(), preds.cpu().numpy()

    def predict_dataframe(self, df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """
        Make predictions for all valid positions in a DataFrame.

        Args:
            df: DataFrame with required feature columns.
            lookback: Lookback window size.

        Returns:
            DataFrame with predictions and probabilities.
        """
        # Check columns
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Apply scaler if available
        features = df[self.feature_columns].values
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Create sequences
        sequences = []
        valid_indices = []

        for i in range(lookback, len(df)):
            seq = features[i - lookback : i]
            sequences.append(seq)
            valid_indices.append(i)

        if not sequences:
            return pd.DataFrame()

        # Predict
        sequences = np.stack(sequences)
        probs, preds = self.predict_batch(sequences)

        # Create result DataFrame
        result = pd.DataFrame(index=df.index[valid_indices])
        result["prediction"] = preds

        for i in range(probs.shape[1]):
            result[f"prob_class_{i}"] = probs[:, i]

        return result


def load_model(
    checkpoint_path: Union[str, Path],
    model_type: str = "transformer",
    device: Optional[torch.device] = None,
) -> BaseClassifier:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint.
        model_type: Model type ("transformer", "lstm").
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config", {})

    model = ModelRegistry.create(
        model_type,
        input_dim=model_config.get("input_dim", 32),
        seq_length=model_config.get("seq_length", 60),
        num_classes=model_config.get("num_classes", 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model.to(device)

    model.eval()
    return model
