#!/bin/bash
# ==============================================================================
# Offshore: Training Script
# ==============================================================================
# Usage: ./scripts/run_train.sh [OPTIONS]
#
# This script runs the full training pipeline:
# 1. Downloads data (if not present)
# 2. Preprocesses data
# 3. Trains the model
# ==============================================================================

set -e  # Exit on error

# Default values
TICKER="AAPL"
MODEL_TYPE="transformer"
START_DATE="2018-01-01"
END_DATE="2024-01-01"
OUTPUT_DIR="outputs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ticker)
            TICKER="$2"
            shift 2
            ;;
        --model)
            MODEL_TYPE="$2"
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
            echo "  --model MODEL         Model type: transformer or lstm (default: transformer)"
            echo "  --start-date DATE     Start date for data (default: 2018-01-01)"
            echo "  --end-date DATE       End date for data (default: 2024-01-01)"
            echo "  --output-dir DIR      Output directory (default: outputs)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up experiment name
EXPERIMENT_NAME="${TICKER}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"

echo "=============================================="
echo "Offshore Training Pipeline"
echo "=============================================="
echo "Ticker:        ${TICKER}"
echo "Model:         ${MODEL_TYPE}"
echo "Date Range:    ${START_DATE} to ${END_DATE}"
echo "Output Dir:    ${EXPERIMENT_DIR}"
echo "=============================================="

# Create output directory
mkdir -p "${EXPERIMENT_DIR}"

# Step 1: Download data
echo ""
echo "[1/3] Downloading data..."
python -m src.cli.main download \
    --ticker "${TICKER}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}"

# Step 2: Preprocess data
echo ""
echo "[2/3] Preprocessing data..."
python -m src.cli.main preprocess \
    --ticker "${TICKER}" \
    --config configs/data.yaml

# Step 3: Train model
echo ""
echo "[3/3] Training model..."
python -m src.cli.main train \
    --model-config "configs/model_${MODEL_TYPE}.yaml" \
    --data-config configs/data.yaml \
    --train-config configs/train.yaml \
    --output-dir "${EXPERIMENT_DIR}"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Results saved to: ${EXPERIMENT_DIR}"
echo "=============================================="
