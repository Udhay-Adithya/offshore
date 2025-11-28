#!/bin/bash
# =============================================================================
# Train Transformer Model
# =============================================================================
# This script trains a Transformer-based classifier for stock trend prediction.
#
# The Transformer model uses:
#   - Multi-head self-attention layers
#   - Positional encoding for sequence ordering
#   - Classification head for trend prediction
#
# Usage: ./train_transformer.sh [TICKER]
# Example: ./train_transformer.sh AAPL
#          ./train_transformer.sh RELIANCE.NS
# =============================================================================

set -e  # Exit on error

# Get ticker from argument or use default
TICKER=${1:-AAPL}

echo "=================================================="
echo "🏛️ Training Transformer Model for: $TICKER"
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
# Train Transformer Model
# -----------------------------------------------------------------------------
echo ""
echo "📋 Configuration:"
echo "   Model Config: configs/model_transformer.yaml"
echo "   Data Config:  configs/data.yaml"
echo "   Train Config: configs/train.yaml"
echo ""

python -m src.cli.main train \
    --model-config configs/model_transformer.yaml \
    --data-config configs/data.yaml \
    --train-config configs/train.yaml \
    --output outputs/transformer_runs \
    --ticker "$TICKER" \
    --data-dir data/processed

echo ""
echo "=================================================="
echo "✅ Transformer training complete!"
echo "=================================================="

# Show output
echo ""
echo "📁 Model saved to: outputs/transformer_runs/"
echo ""
echo "💡 Next steps:"
echo "   1. Evaluate: python -m src.cli.main eval outputs/transformer_runs/${TICKER}_transformer_*/best_model.pt --ticker $TICKER"
echo "   2. Predict:  python -m src.cli.main predict outputs/transformer_runs/${TICKER}_transformer_*/best_model.pt --ticker $TICKER"
