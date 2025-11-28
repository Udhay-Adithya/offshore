#!/bin/bash
# =============================================================================
# Train Both Models for Comparison
# =============================================================================
# This script trains both Transformer and LSTM models on the same stock
# for direct comparison of performance.
#
# Usage: ./train_comparison.sh [TICKER]
# Example: ./train_comparison.sh AAPL
#          ./train_comparison.sh HDFCBANK.NS
# =============================================================================

set -e  # Exit on error

# Get ticker from argument or use default
TICKER=${1:-AAPL}

echo "=================================================="
echo "⚔️  Model Comparison Training for: $TICKER"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Ensure preprocessed data exists
# -----------------------------------------------------------------------------
echo ""
echo "📋 Step 1: Ensuring preprocessed data exists..."

if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "   Running preprocessing..."
    python -m src.cli.main preprocess "$TICKER" \
        --config configs/data.yaml \
        --input data/raw \
        --output data/processed
else
    echo "   ✓ Preprocessed data found"
fi

# -----------------------------------------------------------------------------
# Train Transformer Model
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 2: Training Transformer Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TRANSFORMER_START=$(date +%s)

python -m src.cli.main train \
    --model-config configs/model_transformer.yaml \
    --data-config configs/data.yaml \
    --train-config configs/train.yaml \
    --output outputs/comparison/${TICKER}/transformer \
    --ticker "$TICKER" \
    --data-dir data/processed

TRANSFORMER_END=$(date +%s)
TRANSFORMER_TIME=$((TRANSFORMER_END - TRANSFORMER_START))

echo ""
echo "   ✓ Transformer training completed in ${TRANSFORMER_TIME}s"

# -----------------------------------------------------------------------------
# Train LSTM Model
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 3: Training LSTM Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

LSTM_START=$(date +%s)

python -m src.cli.main train \
    --model-config configs/model_lstm.yaml \
    --data-config configs/data.yaml \
    --train-config configs/train.yaml \
    --output outputs/comparison/${TICKER}/lstm \
    --ticker "$TICKER" \
    --data-dir data/processed

LSTM_END=$(date +%s)
LSTM_TIME=$((LSTM_END - LSTM_START))

echo ""
echo "   ✓ LSTM training completed in ${LSTM_TIME}s"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "✅ Comparison Training Complete!"
echo "=================================================="
echo ""
echo "📊 Training Times:"
echo "   Transformer: ${TRANSFORMER_TIME} seconds"
echo "   LSTM:        ${LSTM_TIME} seconds"
echo ""
echo "📁 Models saved to:"
echo "   outputs/comparison/${TICKER}/transformer/"
echo "   outputs/comparison/${TICKER}/lstm/"
echo ""
echo "💡 Next step: Evaluate both models to compare metrics"
echo ""
echo "   # Evaluate Transformer"
echo "   python -m src.cli.main eval outputs/comparison/${TICKER}/transformer/*/best_model.pt --ticker $TICKER"
echo ""
echo "   # Evaluate LSTM"  
echo "   python -m src.cli.main eval outputs/comparison/${TICKER}/lstm/*/best_model.pt --ticker $TICKER"
