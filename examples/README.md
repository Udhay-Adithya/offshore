# 📚 Offshore Examples

This directory contains example scripts and configurations demonstrating how to use Offshore for stock trend classification.

## 📁 Directory Structure

```bash
examples/
├── README.md                           # This file
├── 01_download_data/
│   ├── download_us_stocks.sh          # Download US stocks (AAPL, MSFT, GOOGL)
│   ├── download_indian_stocks.sh      # Download Indian stocks (NSE/BSE)
│   └── download_indices.sh            # Download market indices
├── 02_preprocess_data/
│   ├── preprocess_single_stock.sh     # Basic preprocessing
│   └── preprocess_multiple_stocks.sh  # Batch preprocessing
├── 03_train_models/
│   ├── train_transformer.sh           # Train Transformer model
│   ├── train_lstm.sh                  # Train LSTM model
│   └── train_comparison.sh            # Train both for comparison
├── 04_evaluate_models/
│   ├── evaluate_model.sh              # Evaluate trained model
│   └── backtest_strategy.sh           # Run backtesting
├── 05_predict/
│   ├── predict_single.sh              # Single stock prediction
│   └── predict_batch.sh               # Batch predictions
├── configs/
│   ├── indian_stocks.yaml             # Config for Indian stocks
│   ├── us_tech.yaml                   # Config for US tech stocks
│   └── aggressive_training.yaml       # Aggressive training settings
└── notebooks/
    └── full_pipeline_example.ipynb    # Complete Jupyter notebook example
```

## 🚀 Quick Start

### 1. Download Data

```bash
cd examples/01_download_data
./download_us_stocks.sh      # For US stocks
./download_indian_stocks.sh  # For Indian stocks (NSE/BSE)
```

### 2. Preprocess Data

```bash
cd examples/02_preprocess_data
./preprocess_single_stock.sh AAPL
```

### 3. Train Model

```bash
cd examples/03_train_models
./train_transformer.sh AAPL
```

### 4. Evaluate

```bash
cd examples/04_evaluate_models
./evaluate_model.sh outputs/AAPL_transformer_*/best_model.pt
```

### 5. Predict

```bash
cd examples/05_predict
./predict_single.sh AAPL outputs/AAPL_transformer_*/best_model.pt
```

## 🇮🇳 Indian Stocks Examples

Indian stocks use special ticker formats:

- **NSE**: Append `.NS` (e.g., `RELIANCE.NS`, `TCS.NS`)
- **BSE**: Append `.BO` (e.g., `RELIANCE.BO`, `TCS.BO`)
- **NIFTY 50**: Use `^NSEI`
- **Sensex**: Use `^BSESN`

See `01_download_data/download_indian_stocks.sh` for examples.

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `offshore download` | Download historical stock data |
| `offshore preprocess` | Clean data, add features, create labels |
| `offshore train` | Train a classification model |
| `offshore eval` | Evaluate model and run backtest |
| `offshore predict` | Make predictions on new data |

## 💡 Tips

1. **Start Simple**: Begin with a single stock before scaling up
2. **Check Data Quality**: Always inspect downloaded data before training
3. **Use Appropriate Lookback**: 60 days works well for daily data
4. **Monitor Training**: Watch for overfitting with validation metrics
5. **Backtest Carefully**: Past performance doesn't guarantee future results

## ⚠️ Disclaimer

These examples are for educational purposes only. Not financial advice!
