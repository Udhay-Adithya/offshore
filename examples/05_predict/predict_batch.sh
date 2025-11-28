#!/bin/bash
# =============================================================================
# Batch Predictions
# =============================================================================
# This script makes predictions for multiple stocks using trained models.
#
# Usage: ./predict_batch.sh
# =============================================================================

set -e  # Exit on error

echo "=================================================="
echo "🔮 Batch Predictions"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Define stocks to predict
# -----------------------------------------------------------------------------

# Add your trained stocks here
STOCKS=("AAPL" "MSFT" "RELIANCE.NS" "TCS.NS" "HDFCBANK.NS")

echo ""
echo "📋 Stocks to predict: ${STOCKS[*]}"
echo ""

# -----------------------------------------------------------------------------
# Make predictions for each stock
# -----------------------------------------------------------------------------

for TICKER in "${STOCKS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔮 Predicting: $TICKER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Find checkpoint for this stock
    CHECKPOINT_PATH=$(find outputs -name "best_model.pt" -path "*${TICKER}*" 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "   ⚠️  No trained model found for $TICKER - skipping"
        echo ""
        continue
    fi
    
    echo "   Using model: $CHECKPOINT_PATH"
    echo ""
    
    python -m src.cli.main predict "$CHECKPOINT_PATH" \
        --ticker "$TICKER" \
        --data-config configs/data.yaml
    
    echo ""
done

echo "=================================================="
echo "✅ Batch predictions complete!"
echo "=================================================="
echo ""
echo "⚠️  DISCLAIMER: These predictions are for educational purposes only."
echo "   NOT financial advice. Do your own research before investing."
