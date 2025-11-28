#!/bin/bash
# =============================================================================
# Single Stock Prediction
# =============================================================================
# This script makes predictions for a single stock using a trained model.
#
# It will:
#   1. Download recent data (if needed)
#   2. Preprocess and create features
#   3. Run inference and output predictions
#
# Usage: ./predict_single.sh [TICKER] [CHECKPOINT_PATH]
# Example: ./predict_single.sh AAPL outputs/AAPL_transformer_*/best_model.pt
#          ./predict_single.sh RELIANCE.NS outputs/RELIANCE.NS_lstm_*/best_model.pt
# =============================================================================

set -e  # Exit on error

# Arguments
TICKER=${1:-AAPL}
CHECKPOINT_PATH=${2:-""}

echo "=================================================="
echo "🔮 Making Predictions for: $TICKER"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Find checkpoint if not provided
# -----------------------------------------------------------------------------
if [ -z "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "🔍 Looking for trained model for $TICKER..."
    
    # Try to find a checkpoint
    CHECKPOINT_PATH=$(find outputs -name "best_model.pt" -path "*${TICKER}*" 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "❌ No checkpoint found for $TICKER"
        echo ""
        echo "Please train a model first:"
        echo "  ./examples/03_train_models/train_transformer.sh $TICKER"
        echo ""
        echo "Or provide a checkpoint path:"
        echo "  ./predict_single.sh $TICKER /path/to/best_model.pt"
        exit 1
    fi
    
    echo "   Found: $CHECKPOINT_PATH"
fi

echo ""
echo "📋 Configuration:"
echo "   Ticker:     $TICKER"
echo "   Checkpoint: $CHECKPOINT_PATH"
echo ""

# -----------------------------------------------------------------------------
# Make Prediction
# -----------------------------------------------------------------------------

python -m src.cli.main predict "$CHECKPOINT_PATH" \
    --ticker "$TICKER" \
    --data-config configs/data.yaml

echo ""
echo "=================================================="
echo "✅ Prediction complete!"
echo "=================================================="
echo ""
echo "⚠️  DISCLAIMER: This prediction is for educational purposes only."
echo "   NOT financial advice. Do your own research before investing."
