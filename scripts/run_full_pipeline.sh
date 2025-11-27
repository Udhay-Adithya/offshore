#!/bin/bash
# ==============================================================================
# Offshore: Full Pipeline Script
# ==============================================================================
# Usage: ./scripts/run_full_pipeline.sh [OPTIONS]
#
# This script runs the complete pipeline:
# 1. Downloads data
# 2. Preprocesses data
# 3. Trains LSTM model
# 4. Trains Transformer model
# 5. Evaluates both models
# 6. Compares results
# ==============================================================================

set -e  # Exit on error

# Default values
TICKER="AAPL"
START_DATE="2018-01-01"
END_DATE="2024-01-01"
OUTPUT_DIR="outputs/experiments"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ticker)
            TICKER="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ticker TICKER       Stock ticker symbol (default: AAPL)"
            echo "  --start-date DATE     Start date for data (default: 2018-01-01)"
            echo "  --end-date DATE       End date for data (default: 2024-01-01)"
            echo "  --output-dir DIR      Output directory (default: outputs/experiments)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${OUTPUT_DIR}/${TICKER}_${TIMESTAMP}"

echo "=============================================="
echo "Offshore Full Pipeline"
echo "=============================================="
echo "Ticker:        ${TICKER}"
echo "Date Range:    ${START_DATE} to ${END_DATE}"
echo "Output Dir:    ${EXPERIMENT_DIR}"
echo "=============================================="

mkdir -p "${EXPERIMENT_DIR}"

# Step 1: Download data
echo ""
echo "[1/6] Downloading data..."
python -m src.cli.main download \
    --ticker "${TICKER}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}"

# Step 2: Preprocess data
echo ""
echo "[2/6] Preprocessing data..."
python -m src.cli.main preprocess \
    --ticker "${TICKER}" \
    --config configs/data.yaml

# Step 3: Train LSTM model
echo ""
echo "[3/6] Training LSTM model..."
LSTM_DIR="${EXPERIMENT_DIR}/lstm"
python -m src.cli.main train \
    --model-config configs/model_lstm.yaml \
    --data-config configs/data.yaml \
    --train-config configs/train.yaml \
    --output-dir "${LSTM_DIR}"

# Step 4: Train Transformer model
echo ""
echo "[4/6] Training Transformer model..."
TRANSFORMER_DIR="${EXPERIMENT_DIR}/transformer"
python -m src.cli.main train \
    --model-config configs/model_transformer.yaml \
    --data-config configs/data.yaml \
    --train-config configs/train.yaml \
    --output-dir "${TRANSFORMER_DIR}"

# Step 5: Evaluate LSTM
echo ""
echo "[5/6] Evaluating LSTM model..."
python -m src.cli.main eval \
    --checkpoint "${LSTM_DIR}/best_model.pt" \
    --data-config configs/data.yaml \
    --eval-config configs/eval.yaml \
    --output-dir "${LSTM_DIR}/eval"

# Step 6: Evaluate Transformer
echo ""
echo "[6/6] Evaluating Transformer model..."
python -m src.cli.main eval \
    --checkpoint "${TRANSFORMER_DIR}/best_model.pt" \
    --data-config configs/data.yaml \
    --eval-config configs/eval.yaml \
    --output-dir "${TRANSFORMER_DIR}/eval"

# Summary
echo ""
echo "=============================================="
echo "Full Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  LSTM:        ${LSTM_DIR}/eval"
echo "  Transformer: ${TRANSFORMER_DIR}/eval"
echo ""
echo "Compare models by viewing the metrics in each eval directory."
echo "=============================================="
