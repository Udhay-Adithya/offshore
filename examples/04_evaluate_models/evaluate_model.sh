#!/bin/bash
# =============================================================================
# Evaluate Trained Model
# =============================================================================
# This script evaluates a trained model on the test set and computes metrics.
#
# Outputs:
#   - Classification metrics (accuracy, F1, precision, recall)
#   - Confusion matrix
#   - Prediction distribution plots
#
# Usage: ./evaluate_model.sh <CHECKPOINT_PATH> [TICKER]
# Example: ./evaluate_model.sh outputs/transformer_runs/AAPL_transformer_*/best_model.pt AAPL
#          ./evaluate_model.sh outputs/lstm_runs/TCS.NS_lstm_*/best_model.pt TCS.NS
# =============================================================================

set -e  # Exit on error

# Check arguments
if [ -z "$1" ]; then
    echo "❌ Error: Please provide checkpoint path"
    echo ""
    echo "Usage: ./evaluate_model.sh <CHECKPOINT_PATH> [TICKER]"
    echo ""
    echo "Example:"
    echo "  ./evaluate_model.sh outputs/AAPL_transformer_20241128/best_model.pt AAPL"
    exit 1
fi

CHECKPOINT_PATH=$1
TICKER=${2:-AAPL}

echo "=================================================="
echo "📊 Evaluating Model"
echo "=================================================="
echo ""
echo "   Checkpoint: $CHECKPOINT_PATH"
echo "   Ticker:     $TICKER"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Run Evaluation
# -----------------------------------------------------------------------------

python -m src.cli.main eval "$CHECKPOINT_PATH" \
    --data-config configs/data.yaml \
    --eval-config configs/eval.yaml \
    --ticker "$TICKER" \
    --data-dir data/processed

echo ""
echo "=================================================="
echo "✅ Evaluation complete!"
echo "=================================================="

# Get output directory (same as checkpoint directory + /eval)
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
OUTPUT_DIR="${CHECKPOINT_DIR}/eval"

echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📋 Generated files:"
ls -la "$OUTPUT_DIR" 2>/dev/null || echo "   (check the output directory)"
echo ""
echo "💡 Review the following files:"
echo "   - metrics.json          : Detailed classification metrics"
echo "   - confusion_matrix.png  : Visual confusion matrix"
echo "   - backtest_results.json : Backtesting performance"
echo "   - equity_curve.png      : Strategy vs Buy & Hold"
