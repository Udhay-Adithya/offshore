# Offshore - Single-Stock Trend Classification

A production-grade, research-quality project for single-stock trend classification (up/down/multi-class) using state-of-the-art time-series deep learning techniques.

## Overview

**Offshore** predicts short-horizon stock trends for a single stock at a time using modern deep learning architectures:

- **Transformer-based classifier** (SOTA time-series architecture)
- **LSTM/GRU baseline** for comparison
- Support for binary (up/down) and multi-class (5-class) trend prediction

### Key Features

- 📊 **Single-stock focus**: Train per-stock models for maximum specialization
- 🔄 **Reproducible**: Config-driven experiments with seed control
- 📈 **Backtesting**: Simple directional strategy evaluation with PnL curves
- 🧩 **Modular**: Easy to extend with new models or features
- 🖥️ **CLI-driven**: Download, preprocess, train, evaluate, and predict from command line

## ⚠️ Disclaimer

**This project is for research and educational purposes only.** It is NOT financial advice and should NOT be used for actual trading decisions. Past performance does not guarantee future results. Always consult qualified financial advisors before making investment decisions.

## Installation

### Prerequisites

- Python 3.11+
- pip or conda

### Install from source

```bash
git clone https://github.com/yourusername/offshore.git
cd offshore
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download data for a stock

```bash
python -m src.cli.main download --ticker AAPL --start_date 2020-01-01 --end_date 2024-01-01 --interval 1d
```

### 2. Preprocess and create features

```bash
python -m src.cli.main preprocess --ticker AAPL --config configs/data.yaml
```

### 3. Train a model

```bash
# Train Transformer model
python -m src.cli.main train \
    --model_config configs/model_transformer.yaml \
    --data_config configs/data.yaml \
    --train_config configs/train.yaml \
    --output_dir outputs/transformer_run1

# Or train LSTM baseline
python -m src.cli.main train \
    --model_config configs/model_lstm.yaml \
    --data_config configs/data.yaml \
    --train_config configs/train.yaml \
    --output_dir outputs/lstm_run1
```

### 4. Evaluate and backtest

```bash
python -m src.cli.main eval \
    --checkpoint_path outputs/transformer_run1/best_model.pt \
    --data_config configs/data.yaml \
    --eval_config configs/eval.yaml \
    --output_dir outputs/transformer_run1/eval
```

### 5. Make predictions

```bash
python -m src.cli.main predict \
    --checkpoint_path outputs/transformer_run1/best_model.pt \
    --ticker AAPL \
    --data_config configs/data.yaml
```

## Project Structure

```
offshore/
├── configs/                    # Configuration files
│   ├── data.yaml              # Data paths, tickers, lookback/horizon
│   ├── model_base.yaml        # Default model hyperparameters
│   ├── model_transformer.yaml # Transformer-specific config
│   ├── model_lstm.yaml        # LSTM-specific config
│   ├── train.yaml             # Training hyperparameters
│   └── eval.yaml              # Evaluation and backtest settings
├── data/
│   ├── raw/                   # Raw downloaded OHLCV data
│   └── processed/             # Processed features and labels
├── notebooks/                 # Exploration and experiment notebooks
├── src/
│   ├── config/                # Configuration loading utilities
│   ├── data/                  # Data download, preprocessing, datasets
│   ├── features/              # Technical indicators and labeling
│   ├── models/                # Neural network architectures
│   ├── training/              # Training loop and callbacks
│   ├── evaluation/            # Metrics, backtesting, plotting
│   ├── cli/                   # Command-line interface
│   └── inference/             # Model loading and prediction
├── tests/                     # Unit tests
├── scripts/                   # Utility scripts
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Configuration

All experiments are configured via YAML files in `configs/`:

### data.yaml

```yaml
ticker: AAPL
interval: "1d"
lookback: 60        # Number of past bars as input
horizon: 5          # Predict trend over next H bars
label_type: binary  # "binary" or "multiclass"
```

### model_transformer.yaml

```yaml
model_type: transformer
input_dim: 32       # Number of input features
d_model: 64
nhead: 4
num_layers: 3
dim_feedforward: 256
dropout: 0.1
num_classes: 2
```

### train.yaml

```yaml
batch_size: 64
epochs: 100
learning_rate: 0.001
optimizer: adamw
weight_decay: 0.01
scheduler: cosine
early_stopping_patience: 10
```

## Training a New Stock

To train on a different stock:

1. Update `configs/data.yaml` with the new ticker
2. Download data: `python -m src.cli.main download --ticker NEW_TICKER ...`
3. Preprocess: `python -m src.cli.main preprocess --ticker NEW_TICKER ...`
4. Train: `python -m src.cli.main train ...`

The model is trained from scratch on the new stock's historical data.

## Model Architectures

### Time-Series Transformer

- Encoder-only Transformer with positional encoding
- Multi-head self-attention over time dimension
- Channel attention (SE block) for feature importance
- CLS token pooling → MLP classification head

### LSTM/GRU Baseline

- Bidirectional LSTM/GRU encoder
- Final hidden state → MLP classification head
- Configurable depth and hidden size

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Backtest**: Cumulative returns, Max drawdown, Win rate, Sharpe ratio (approximate)

## Limitations

1. **Single-stock models**: No cross-stock transfer learning
2. **No transaction costs**: Backtest approximates but doesn't model real costs
3. **No real-time data**: Uses historical data only
4. **Research only**: Not validated for live trading

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit PRs.

## License

MIT License - see LICENSE file for details.

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Temporal Fusion Transformers (Lim et al., 2021)
- N-BEATS (Oreshkin et al., 2019)
