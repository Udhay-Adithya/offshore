#!/bin/bash
# =============================================================================
# Train LSTM Model
# =============================================================================
# This script trains an LSTM-based classifier for stock trend prediction.
#
# The LSTM model uses:
#   - Bidirectional LSTM layers (optional)
#   - Dropout for regularization
#   - Classification head for trend prediction
#
# LSTM is often a good baseline and can be faster to train than Transformers.
#
# Usage: ./train_lstm.sh [TICKER]
# Example: ./train_lstm.sh AAPL
#          ./train_lstm.sh TCS.NS
# =============================================================================

set -e  # Exit on error

# Get ticker from argument or use default
TICKER=${1:-AAPL}

echo "=================================================="
echo "🔄 Training LSTM Model for: $TICKER"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Check if preprocessed data exists
# -----------------------------------------------------------------------------
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "⚠️  No preprocessed data found. Running preprocessing first..."
    python -m src.cli.main preprocess "$TICKER" \
        --config configs/data.yaml \
        --input data/raw \
        --output data/processed
fi

# -----------------------------------------------------------------------------
# Train LSTM Model
# -----------------------------------------------------------------------------
echo ""
echo "📋 Configuration:"
echo "   Model Config: configs/model_lstm.yaml"
echo "   Data Config:  configs/data.yaml"
echo "   Train Config: configs/train.yaml"
echo ""

python -m src.cli.main train \
    --model-config configs/model_lstm.yaml \
    --data-config configs/data.yaml \
    --train-config configs/train.yaml \
    --output outputs/lstm_runs \
    --ticker "$TICKER" \
    --data-dir data/processed

echo ""
echo "=================================================="
echo "✅ LSTM training complete!"
echo "=================================================="

# Show output
echo ""
echo "📁 Model saved to: outputs/lstm_runs/"
echo ""
echo "💡 Next steps:"
echo "   1. Evaluate: python -m src.cli.main eval outputs/lstm_runs/${TICKER}_lstm_*/best_model.pt --ticker $TICKER"
echo "   2. Predict:  python -m src.cli.main predict outputs/lstm_runs/${TICKER}_lstm_*/best_model.pt --ticker $TICKER"
