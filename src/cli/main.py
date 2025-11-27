"""
Command-line interface for Offshore.

Provides subcommands for downloading, preprocessing, training, evaluation, and prediction.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("offshore")

# Create Typer app
app = typer.Typer(
    name="offshore",
    help="Single-stock trend classification using deep learning",
    add_completion=False,
)

console = Console()


@app.command()
def download(
    ticker: str = typer.Argument(..., help="Stock ticker symbol (e.g., AAPL)"),
    start_date: str = typer.Option("2015-01-01", "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2024-01-01", "--end", "-e", help="End date (YYYY-MM-DD)"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval (1d, 1h, etc.)"),
    output_dir: str = typer.Option("data/raw", "--output", "-o", help="Output directory"),
) -> None:
    """Download historical stock data for a ticker."""
    from src.data.download import download_data

    console.print(f"[bold blue]Downloading {ticker} data...[/bold blue]")
    console.print(f"  Date range: {start_date} to {end_date}")
    console.print(f"  Interval: {interval}")

    try:
        df = download_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            output_dir=output_dir,
        )

        console.print(f"[bold green]✓ Downloaded {len(df)} rows[/bold green]")
        console.print(f"  Saved to: {output_dir}/{ticker}_{interval}.csv")
        console.print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def preprocess(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    config: str = typer.Option("configs/data.yaml", "--config", "-c", help="Data config file"),
    input_dir: str = typer.Option("data/raw", "--input", "-i", help="Input directory"),
    output_dir: str = typer.Option("data/processed", "--output", "-o", help="Output directory"),
    interval: str = typer.Option("1d", "--interval", help="Data interval"),
) -> None:
    """Preprocess raw data: clean, add features, create labels, split."""
    from src.config import load_config
    from src.data.download import DataDownloader
    from src.data.preprocess import DataPreprocessor

    console.print(f"[bold blue]Preprocessing {ticker} data...[/bold blue]")

    try:
        # Load config
        data_config = load_config(config)

        # Load raw data
        downloader = DataDownloader(output_dir=input_dir)
        df = downloader.load(ticker, interval)
        console.print(f"  Loaded {len(df)} rows from raw data")

        # Preprocess
        preprocessor = DataPreprocessor(data_config)
        train_df, val_df, test_df = preprocessor.process(df)

        # Save
        preprocessor.save_processed(train_df, val_df, test_df, ticker, output_dir)

        console.print(f"[bold green]✓ Preprocessing complete[/bold green]")
        console.print(f"  Train samples: {len(train_df)}")
        console.print(f"  Val samples: {len(val_df)}")
        console.print(f"  Test samples: {len(test_df)}")
        console.print(f"  Features: {len(preprocessor.feature_columns)}")
        console.print(f"  Saved to: {output_dir}/")

    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def train(
    model_config: str = typer.Option(
        "configs/model_transformer.yaml", "--model-config", "-m", help="Model configuration file"
    ),
    data_config: str = typer.Option(
        "configs/data.yaml", "--data-config", "-d", help="Data configuration file"
    ),
    train_config: str = typer.Option(
        "configs/train.yaml", "--train-config", "-t", help="Training configuration file"
    ),
    output_dir: str = typer.Option(
        "outputs/run", "--output", "-o", help="Output directory for checkpoints and logs"
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker override"),
    data_dir: str = typer.Option("data/processed", "--data-dir", help="Processed data directory"),
) -> None:
    """Train a model on preprocessed data."""
    import torch
    from src.config import load_config, merge_configs
    from src.data.preprocess import load_processed_data
    from src.data.dataset import create_dataloaders
    from src.models import ModelRegistry
    from src.training import Trainer

    console.print("[bold blue]Starting training...[/bold blue]")

    try:
        # Load configs
        model_cfg = load_config(model_config)
        data_cfg = load_config(data_config)
        train_cfg = load_config(train_config)

        # Merge configs
        config = merge_configs(merge_configs(data_cfg, model_cfg), train_cfg)

        # Get ticker from config or argument
        ticker = ticker or data_cfg.get("ticker", "AAPL")
        console.print(f"  Ticker: {ticker}")

        # Load processed data
        train_df, val_df, test_df, metadata = load_processed_data(ticker, data_dir)
        feature_columns = metadata["feature_columns"]

        console.print(f"  Train samples: {len(train_df)}")
        console.print(f"  Val samples: {len(val_df)}")
        console.print(f"  Features: {len(feature_columns)}")

        # Create dataloaders
        lookback = data_cfg.get("features", {}).get("lookback", 60)
        batch_size = train_cfg.get("training", {}).get("batch_size", 64)

        train_loader, val_loader, _ = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=None,
            feature_columns=feature_columns,
            lookback=lookback,
            batch_size=batch_size,
            augmentation=data_cfg.get("augmentation"),
        )

        # Create model
        model_type = model_cfg.get("model_type", "transformer")
        num_classes = 2 if data_cfg.get("labeling", {}).get("label_type") == "binary" else 5

        # Prepare model kwargs based on type
        if model_type == "transformer":
            transformer_cfg = model_cfg.get("transformer", {})
            model_kwargs = {
                "input_dim": len(feature_columns),
                "seq_length": lookback,
                "num_classes": num_classes,
                "d_model": transformer_cfg.get("d_model", 64),
                "nhead": transformer_cfg.get("nhead", 4),
                "num_layers": transformer_cfg.get("num_layers", 3),
                "dim_feedforward": transformer_cfg.get("dim_feedforward", 256),
                "dropout": transformer_cfg.get("dropout", 0.1),
                "pos_encoding": transformer_cfg.get("pos_encoding", "sinusoidal"),
                "pooling": transformer_cfg.get("pooling", "cls"),
            }
        else:  # lstm or gru
            lstm_cfg = model_cfg.get("lstm", {})
            model_kwargs = {
                "input_dim": len(feature_columns),
                "seq_length": lookback,
                "num_classes": num_classes,
                "hidden_dim": lstm_cfg.get("hidden_dim", 128),
                "num_layers": lstm_cfg.get("num_layers", 2),
                "bidirectional": lstm_cfg.get("bidirectional", True),
                "dropout": lstm_cfg.get("dropout", 0.2),
                "use_gru": lstm_cfg.get("use_gru", False),
                "pooling": lstm_cfg.get("pooling", "last"),
            }

        model = ModelRegistry.create(model_type, **model_kwargs)
        console.print(f"  Model: {model_type}")
        console.print(f"  Parameters: {model.get_num_parameters():,}")

        # Create output directory with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"{ticker}_{model_type}_{timestamp}"

        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=output_path,
        )

        history = trainer.train()

        console.print(f"\n[bold green]✓ Training complete[/bold green]")
        console.print(f"  Best Val F1: {max(history['val_f1']):.4f}")
        console.print(f"  Output: {output_path}")

    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("eval")
def evaluate(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    data_config: str = typer.Option(
        "configs/data.yaml", "--data-config", "-d", help="Data configuration file"
    ),
    eval_config: str = typer.Option(
        "configs/eval.yaml", "--eval-config", "-e", help="Evaluation configuration file"
    ),
    output_dir: str = typer.Option(
        None, "--output", "-o", help="Output directory (default: same as checkpoint)"
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker override"),
    data_dir: str = typer.Option("data/processed", "--data-dir", help="Processed data directory"),
) -> None:
    """Evaluate a trained model and run backtest."""
    import torch
    import numpy as np
    from src.config import load_config
    from src.data.preprocess import load_processed_data
    from src.data.dataset import create_dataloaders
    from src.models import ModelRegistry
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.backtest import run_backtest
    from src.evaluation.plots import (
        plot_confusion_matrix,
        plot_equity_curve,
        plot_prediction_distribution,
    )

    console.print("[bold blue]Evaluating model...[/bold blue]")

    try:
        # Load configs
        data_cfg = load_config(data_config)
        eval_cfg = load_config(eval_config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint.get("model_config", {})

        # Get ticker
        ticker = ticker or data_cfg.get("ticker", "AAPL")

        # Load processed data
        train_df, val_df, test_df, metadata = load_processed_data(ticker, data_dir)
        feature_columns = metadata["feature_columns"]

        console.print(f"  Test samples: {len(test_df)}")

        # Create test dataloader
        lookback = data_cfg.get("features", {}).get("lookback", 60)

        _, _, test_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_columns=feature_columns,
            lookback=lookback,
            batch_size=64,
        )

        # Determine model type from checkpoint path or config
        checkpoint_dir = Path(checkpoint_path).parent
        if "transformer" in str(checkpoint_path).lower():
            model_type = "transformer"
        elif "lstm" in str(checkpoint_path).lower():
            model_type = "lstm"
        else:
            model_type = "transformer"  # Default

        # Recreate model
        num_classes = model_config.get("num_classes", 2)
        model = ModelRegistry.create(
            model_type,
            input_dim=model_config.get("input_dim", len(feature_columns)),
            seq_length=model_config.get("seq_length", lookback),
            num_classes=num_classes,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Run inference
        all_predictions = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=-1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probs = np.array(all_probs)

        # Compute metrics
        calculator = MetricsCalculator(num_classes=num_classes)
        metrics = calculator.compute(targets, predictions, probs)
        calculator.print_summary(metrics)

        # Setup output directory
        if output_dir is None:
            output_dir = checkpoint_dir / "eval"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(output_path / "metrics.json", "w") as f:
            # Convert numpy types for JSON serialization
            metrics_json = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            metrics_json["confusion_matrix"] = metrics["confusion_matrix"]
            json.dump(metrics_json, f, indent=2, default=str)

        # Plot confusion matrix
        class_names = ["down", "up"] if num_classes == 2 else None
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            class_names=class_names,
            save_path=output_path / "confusion_matrix.png",
        )

        # Run backtest
        backtest_cfg = eval_cfg.get("backtest", {})
        if backtest_cfg.get("enabled", True):
            console.print("\n[bold blue]Running backtest...[/bold blue]")

            # Get prices from test data
            prices = test_df["close"].values[lookback:]

            result = run_backtest(
                prices=prices, predictions=predictions, probabilities=probs, config=backtest_cfg
            )

            result.print_summary()

            # Save backtest results
            with open(output_path / "backtest_results.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Plot equity curve
            plot_equity_curve(
                result.equity_curve,
                benchmark=prices / prices[0] * result.equity_curve[0],
                save_path=output_path / "equity_curve.png",
            )

        # Plot prediction distribution
        plot_prediction_distribution(
            targets, predictions, probs, save_path=output_path / "prediction_distribution.png"
        )

        console.print(f"\n[bold green]✓ Evaluation complete[/bold green]")
        console.print(f"  Results saved to: {output_path}")

    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def predict(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to predict"),
    data_config: str = typer.Option(
        "configs/data.yaml", "--data-config", "-d", help="Data configuration file"
    ),
    recent_data: str = typer.Option(
        None,
        "--recent-data",
        "-r",
        help="Path to recent data CSV (optional, will download if not provided)",
    ),
) -> None:
    """Make predictions using a trained model."""
    import torch
    import numpy as np
    from datetime import datetime, timedelta
    from src.config import load_config
    from src.data.download import download_data
    from src.data.preprocess import DataPreprocessor
    from src.models import ModelRegistry

    console.print("[bold blue]Making predictions...[/bold blue]")

    try:
        # Load config and checkpoint
        data_cfg = load_config(data_config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint.get("model_config", {})

        ticker = ticker or data_cfg.get("ticker", "AAPL")
        lookback = data_cfg.get("features", {}).get("lookback", 60)

        # Get recent data
        if recent_data:
            import pandas as pd

            df = pd.read_csv(recent_data, index_col=0, parse_dates=True)
        else:
            # Download recent data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            df = download_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=data_cfg.get("interval", "1d"),
                save=False,
            )

        console.print(f"  Loaded {len(df)} rows of data")

        # Preprocess (without splitting)
        preprocessor = DataPreprocessor(data_cfg)
        df = preprocessor._clean_data(df)
        df = preprocessor._add_features(df)

        feature_columns = preprocessor.feature_columns

        # Load scaler if available
        import joblib

        scaler_path = Path("data/processed") / f"{ticker}_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            df[feature_columns] = scaler.transform(df[feature_columns])

        # Get last lookback rows
        if len(df) < lookback:
            console.print(f"[red]Error: Need at least {lookback} rows, got {len(df)}[/red]")
            raise typer.Exit(code=1)

        sequence = df[feature_columns].values[-lookback:]
        sequence = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0)

        # Determine model type
        if "transformer" in str(checkpoint_path).lower():
            model_type = "transformer"
        else:
            model_type = "lstm"

        # Create and load model
        num_classes = model_config.get("num_classes", 2)
        model = ModelRegistry.create(
            model_type,
            input_dim=model_config.get("input_dim", len(feature_columns)),
            seq_length=model_config.get("seq_length", lookback),
            num_classes=num_classes,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Predict
        with torch.no_grad():
            output = model(sequence)
            probs = torch.softmax(output, dim=-1).numpy()[0]
            prediction = output.argmax(dim=-1).item()

        # Display results
        horizon = data_cfg.get("labeling", {}).get("horizon", 5)
        class_names = (
            ["DOWN", "UP"]
            if num_classes == 2
            else ["STRONG_DOWN", "MILD_DOWN", "FLAT", "MILD_UP", "STRONG_UP"]
        )

        console.print("\n[bold green]Prediction Results[/bold green]")
        console.print(f"  Ticker: {ticker}")
        console.print(f"  Last Price: ${df['close'].iloc[-1]:.2f}")
        console.print(f"  Prediction Date: {df.index[-1]}")
        console.print(f"  Horizon: {horizon} bars")
        console.print(f"\n  [bold]Predicted Trend: {class_names[prediction]}[/bold]")
        console.print(f"\n  Class Probabilities:")
        for i, name in enumerate(class_names):
            bar_length = int(probs[i] * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            console.print(f"    {name:12s}: {bar} {probs[i]:.1%}")

    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Show package information and configuration."""
    from src import __version__

    console.print("[bold]Offshore - Stock Trend Classification[/bold]")
    console.print(f"Version: {__version__}")
    console.print("\nAvailable commands:")
    console.print("  download   - Download historical stock data")
    console.print("  preprocess - Preprocess data and create features")
    console.print("  train      - Train a model")
    console.print("  eval       - Evaluate model and run backtest")
    console.print("  predict    - Make predictions with trained model")
    console.print("\nExample workflow:")
    console.print("  1. offshore download AAPL --start 2015-01-01 --end 2024-01-01")
    console.print("  2. offshore preprocess AAPL")
    console.print("  3. offshore train --model-config configs/model_transformer.yaml")
    console.print("  4. offshore eval outputs/run/best_model.pt")
    console.print("  5. offshore predict outputs/run/best_model.pt --ticker AAPL")


if __name__ == "__main__":
    app()
