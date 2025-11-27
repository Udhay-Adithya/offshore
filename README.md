<p align="center">
  <img src="assets/offshore_logo.png" width="180" alt="Offshore Logo">
</p>

<h1 align="center">Offshore</h1>
<h3 align="center">Production-Grade Single-Stock Trend Classification with Deep Learning</h3>

<p align="center">
  <a href="https://github.com/Udhay-Adithya/offshore/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"></a>
  <a href="https://github.com/Udhay-Adithya/offshore/stargazers"><img src="https://img.shields.io/github/stars/Udhay-Adithya/offshore?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-model-zoo">Models</a> •
  <a href="#-documentation">Docs</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🔥 News

- 🚀 **Nov 2025**: Initial release with Transformer and LSTM classifiers
- 📊 **Nov 2025**: Full support for Indian stocks (NSE/BSE) via yfinance
- 🧪 **Nov 2025**: Complete test suite with pytest fixtures

---

## ⚠️ Disclaimer

> **This project is for research and educational purposes only.** It is NOT financial advice and should NOT be used for actual trading or investment decisions. The authors are not responsible for any financial losses. Past performance does not guarantee future results. Always consult qualified financial advisors.

---

## 📖 Introduction

**Offshore** is a production-grade framework for single-stock trend classification using state-of-the-art deep learning. It predicts short-horizon price movements (up/down or multi-class) for stocks worldwide, including **Indian markets** (NSE/BSE).

The project features:

- 🏛️ **Transformer Classifier** — SOTA attention-based architecture for time series
- 🔄 **LSTM/GRU Baseline** — Strong sequential model for comparison
- 🌍 **Global Stock Support** — Works with US, Indian, European, and Asian markets

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Single-Stock Focus** | Train specialized models per stock for maximum accuracy |
| 🔧 **Config-Driven** | All hyperparameters managed via YAML configs |
| 📊 **Technical Indicators** | Built-in RSI, MACD, Bollinger Bands, MA, and more |
| 🏷️ **Flexible Labeling** | Binary (up/down) or 5-class trend classification |
| 📈 **Backtesting** | Simple directional strategy with PnL tracking |
| 🖥️ **CLI Interface** | Download, preprocess, train, eval, predict from terminal |
| 🧪 **Well-Tested** | Comprehensive pytest suite with fixtures |
| 🌏 **Multi-Market** | US, India (NSE/BSE), Europe, Asia support |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Udhay-Adithya/offshore.git
cd offshore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Basic Workflow

```bash
# 1️⃣ Download stock data
python -m src.cli.main download \\
    --ticker AAPL \\
    --start_date 2020-01-01 \\
    --end_date 2024-01-01

# 2️⃣ Preprocess and create features
python -m src.cli.main preprocess \\
    --ticker AAPL \\
    --config configs/data.yaml

# 3️⃣ Train Transformer model
python -m src.cli.main train \\
    --model_config configs/model_transformer.yaml \\
    --data_config configs/data.yaml \\
    --train_config configs/train.yaml \\
    --output_dir outputs/aapl_transformer

# 4️⃣ Evaluate and backtest
python -m src.cli.main eval \\
    --checkpoint_path outputs/aapl_transformer/best_model.pt \\
    --eval_config configs/eval.yaml \\
    --output_dir outputs/aapl_transformer/eval

# 5️⃣ Make predictions
python -m src.cli.main predict \\
    --checkpoint_path outputs/aapl_transformer/best_model.pt \\
    --ticker AAPL
```

---

## 🏗️ Model Zoo

### Available Architectures

| Model | Type | Parameters | Best For |
|-------|------|------------|----------|
| **TransformerClassifier** | Attention-based | ~500K | Capturing long-range dependencies |
| **LSTMClassifier** | Recurrent | ~300K | Sequential patterns, baseline |

### Model Configurations

<details>
<summary>📁 Transformer Config (click to expand)</summary>

```yaml
# configs/model_transformer.yaml
model:
  type: transformer
  input_size: 32
  d_model: 128
  nhead: 8
  num_encoder_layers: 4
  dim_feedforward: 512
  dropout: 0.1
  num_classes: 2
  max_seq_length: 100
```

</details>

<details>
<summary>📁 LSTM Config (click to expand)</summary>

```yaml
# configs/model_lstm.yaml
model:
  type: lstm
  input_size: 32
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: false
  num_classes: 2
```

</details>

---

## 📁 Project Structure

```bash
offshore/
├── configs/                    # 📋 YAML configuration files
│   ├── data.yaml              # Data paths, tickers, lookback/horizon
│   ├── model_transformer.yaml # Transformer hyperparameters
│   ├── model_lstm.yaml        # LSTM hyperparameters
│   ├── train.yaml             # Training settings
│   └── eval.yaml              # Evaluation settings
├── data/
│   ├── raw/                   # 📥 Downloaded OHLCV data
│   └── processed/             # 🔧 Processed features
├── src/
│   ├── config/                # Configuration loading
│   ├── data/                  # Download, preprocess, datasets
│   ├── features/              # Technical indicators, labeling
│   ├── models/                # Neural network architectures
│   ├── training/              # Training loop, callbacks
│   ├── evaluation/            # Metrics, backtesting, plots
│   ├── cli/                   # Command-line interface
│   └── inference/             # Model inference
├── tests/                     # 🧪 Unit tests
├── scripts/                   # 🔧 Utility scripts
├── notebooks/                 # 📓 Jupyter notebooks
├── CONTRIBUTING.md            # Contribution guidelines
├── CODE_OF_CONDUCT.md         # Community guidelines
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## 📊 Technical Indicators

Built-in features computed automatically:

| Category | Indicators |
|----------|------------|
| **Returns** | Simple returns, Log returns |
| **Moving Averages** | SMA(5,10,20,50), EMA(5,10,20,50) |
| **Momentum** | RSI(14), MACD, Rate of Change |
| **Volatility** | Bollinger Bands, ATR, Rolling Std |
| **Volume** | Volume Ratio, VWAP |

---

## 🏷️ Labeling Schemes

### Binary Classification

| Label | Condition | Value |
|-------|-----------|-------|
| Up | Future return > 0 | 1 |
| Down | Future return ≤ 0 | 0 |

### Multi-class Classification (5 classes)

| Label | Condition | Value |
|-------|-----------|-------|
| Strong Down | Return < -2% | 0 |
| Mild Down | -2% ≤ Return < -0.5% | 1 |
| Flat | -0.5% ≤ Return < 0.5% | 2 |
| Mild Up | 0.5% ≤ Return < 2% | 3 |
| Strong Up | Return ≥ 2% | 4 |

---

## 📋 Configuration Reference

<details>
<summary>📁 data.yaml</summary>

```yaml
data:
  ticker: AAPL
  interval: "1d"
  lookback_window: 60
  prediction_horizon: 5
  label_type: binary  # or "multiclass"
  train_end_date: "2022-12-31"
  val_end_date: "2023-06-30"
```

</details>

<details>
<summary>📁 train.yaml</summary>

```yaml
training:
  seed: 42
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.01
  scheduler: cosine
  early_stopping_patience: 15
  gradient_clip_norm: 1.0
```

</details>

---

## 🧪 Testing

```bash

# Run all tests

pytest tests/ -v

# Run with coverage

pytest tests/ --cov=src --cov-report=html

# Run specific module tests

pytest tests/test_models.py -v
pytest tests/test_data.py -v
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall prediction accuracy |
| **Precision** | True positives / Predicted positives |
| **Recall** | True positives / Actual positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve (binary) |
| **Confusion Matrix** | Detailed breakdown by class |

### Backtest Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative PnL percentage |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Sharpe Ratio** | Risk-adjusted returns |
| **Win Rate** | Percentage of profitable trades |

---

## 🗺️ Roadmap

- [ ] 🔄 Add TCN (Temporal Convolutional Network) model
- [ ] 🧠 Integrate N-BEATS and Informer architectures
- [ ] 📰 Add sentiment features from news/social media
- [ ] 🔗 Multi-stock transfer learning
- [ ] 🚀 Real-time prediction API
- [ ] 🎛️ Hyperparameter optimization (Optuna)
- [ ] 💹 Advanced backtesting with transaction costs
- [ ] 🐳 Docker containerization

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your changes (\`git commit -m 'feat: Add AmazingFeature'\`)
4. Push to the branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References & Resources

This project was built with insights from the following research papers, tutorials, and open-source projects:

### 📄 Research Papers

| Paper | Topic |
|-------|-------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original Transformer architecture |
| [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363) | Time series transformers |
| [Stock Movement Prediction with Transformers](https://www.sciencedirect.com/science/article/abs/pii/S0957417422006170) | Financial ML |
| [Deep Learning for Stock Prediction](https://www.sciencedirect.com/science/article/abs/pii/S0957417422013100) | Stock trend classification |
| [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | Transformer effectiveness study |
| [Crossformer: Transformer with Cross-Dimension Dependency](https://arxiv.org/pdf/2306.02136.pdf) | Advanced time series transformer |
| [PatchTST](https://arxiv.org/pdf/2211.14730.pdf) | Patching for time series |
| [Informer: Beyond Efficient Transformer](https://arxiv.org/pdf/2012.07436.pdf) | Efficient long sequence modeling |
| [Time Series Library Benchmarks](https://arxiv.org/pdf/2303.06286.pdf) | Comprehensive benchmarks |
| [Autoformer](https://arxiv.org/pdf/2106.13008.pdf) | Auto-correlation for time series |
| [FEDformer](https://arxiv.org/pdf/2201.12740.pdf) | Frequency enhanced decomposition |
| [TimesNet](https://arxiv.org/pdf/2210.02186.pdf) | Temporal 2D-variation modeling |
| [iTransformer](https://arxiv.org/pdf/2310.06625.pdf) | Inverted Transformer for forecasting |
| [Stock Prediction Survey](https://arxiv.org/abs/2408.12408) | Comprehensive survey |
| [Transformer for Time Series Regression](https://pub.towardsai.net/time-series-regression-using-transformer-models-a-plain-english-introduction-3215892e1cc) | Plain English introduction |

### 🔧 Project Structure & MLOps

| Resource | Description |
|----------|-------------|
| [ML Project Structure Guide](https://dev.to/luxdevhq/generic-folder-structure-for-your-machine-learning-projects-4coe) | Generic folder structure |
| [ML Project Structure Demo](https://github.com/kylebradbury/ml-project-structure-demo) | Example project layout |
| [Python Repo Structure for Production](https://python.plainenglish.io/making-python-code-repo-well-structured-for-production-mlops-1-fbc2342a19d5) | Production-ready structure |
| [ML Project with MLOps in Mind](https://towardsdatascience.com/structuring-your-machine-learning-project-with-mlops-in-mind-41a8d65987c9/) | MLOps best practices |
| [MLOps Pipeline Development](https://blog.devops.dev/master-machine-learning-pipeline-development-mlops-project-1-project-structure-setup-part-3-618ad96560fa) | Pipeline setup guide |

### 🛠️ Code References & Implementations

| Repository | Description |
|------------|-------------|
| [Time-Series-Transformer-Pytorch](https://github.com/ctxj/Time-Series-Transformer-Pytorch) | PyTorch transformer for time series |
| [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) | Comprehensive time series library |
| [Transformer-based-Stock-Prediction](https://github.com/mrunalmania/Transformer-based-Stock-Prediction) | Stock prediction with transformers |
| [ML Project Templates](https://github.com/topics/machine-learning-project-template) | Various ML project templates |
| [Time Series Classification](https://github.com/topics/time-series-classification) | Time series classification resources |

### 📖 Tutorials & Guides

| Tutorial | Topic |
|----------|-------|
| [PyTorch Transformer for Time Series](https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e/) | Step-by-step guide |
| [ML in Python Step by Step](https://www.machinelearningmastery.com/machine-learning-in-python-step-by-step/) | Comprehensive Python ML guide |
| [GeeksforGeeks ML Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/) | ML project ideas |
| [ProjectPro ML Projects](https://www.projectpro.io/article/top-10-machine-learning-projects-for-beginners-in-2021/397) | Beginner ML projects |

---

## 🙏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) — Free financial data API
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [scikit-learn](https://scikit-learn.org/) — Machine learning utilities
- [pandas](https://pandas.pydata.org/) — Data manipulation

---

## 📞 Contact

**Author**: Udhay Adithya J  
**Email**: <udhayxd@gmail.com>

- 🐛 **Issues**: [GitHub Issues](https://github.com/Udhay-Adithya/offshore/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Udhay-Adithya/offshore/discussions)
- 🌐 **Homepage**: [https://github.com/Udhay-Adithya/offshore](https://github.com/Udhay-Adithya/offshore)

---

<p align="center">
  <b>⭐ Star this repo if you find it useful!</b>
</p>
