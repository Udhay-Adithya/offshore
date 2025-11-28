#!/bin/bash
# =============================================================================
# Run Backtesting Strategy
# =============================================================================
# This script focuses on backtesting a trained model's predictions.
# It simulates trading based on model predictions and compares
# against a buy-and-hold benchmark.
#
# The backtest:
#   - Goes long when model predicts UP
#   - Goes to cash (or short) when model predicts DOWN
#   - Tracks equity curve and calculates returns
#
# Usage: ./backtest_strategy.sh <CHECKPOINT_PATH> [TICKER]
# Example: ./backtest_strategy.sh outputs/AAPL_transformer_*/best_model.pt AAPL
# =============================================================================

set -e  # Exit on error

# Check arguments
if [ -z "$1" ]; then
    echo "❌ Error: Please provide checkpoint path"
    echo ""
    echo "Usage: ./backtest_strategy.sh <CHECKPOINT_PATH> [TICKER]"
    exit 1
fi

CHECKPOINT_PATH=$1
TICKER=${2:-AAPL}

echo "=================================================="
echo "📈 Running Backtest Strategy"
echo "=================================================="
echo ""
echo "   Checkpoint: $CHECKPOINT_PATH"
echo "   Ticker:     $TICKER"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Run Evaluation with Backtest Enabled
# -----------------------------------------------------------------------------

# The eval command includes backtesting by default (configs/eval.yaml)
python -m src.cli.main eval "$CHECKPOINT_PATH" \
    --data-config configs/data.yaml \
    --eval-config configs/eval.yaml \
    --ticker "$TICKER" \
    --data-dir data/processed

echo ""
echo "=================================================="
echo "✅ Backtest complete!"
echo "=================================================="

# Get output directory
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
OUTPUT_DIR="${CHECKPOINT_DIR}/eval"

echo ""
echo "📊 Backtest Results:"
echo ""

# Display backtest results if available
if [ -f "$OUTPUT_DIR/backtest_results.json" ]; then
    echo "   Strategy Performance (from backtest_results.json):"
    cat "$OUTPUT_DIR/backtest_results.json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"   Total Return:     {data.get('total_return', 'N/A'):.2%}\" if isinstance(data.get('total_return'), (int, float)) else f\"   Total Return: {data.get('total_return', 'N/A')}\")
print(f\"   Sharpe Ratio:     {data.get('sharpe_ratio', 'N/A'):.2f}\" if isinstance(data.get('sharpe_ratio'), (int, float)) else f\"   Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}\")
print(f\"   Max Drawdown:     {data.get('max_drawdown', 'N/A'):.2%}\" if isinstance(data.get('max_drawdown'), (int, float)) else f\"   Max Drawdown: {data.get('max_drawdown', 'N/A')}\")
print(f\"   Win Rate:         {data.get('win_rate', 'N/A'):.2%}\" if isinstance(data.get('win_rate'), (int, float)) else f\"   Win Rate: {data.get('win_rate', 'N/A')}\")
print(f\"   Total Trades:     {data.get('total_trades', 'N/A')}\")
" 2>/dev/null || echo "   (Unable to parse backtest_results.json)"
else
    echo "   ⚠️  backtest_results.json not found"
fi

echo ""
echo "📁 Plots saved to: $OUTPUT_DIR"
echo "   - equity_curve.png : Strategy vs Buy & Hold comparison"
echo ""
echo "⚠️  DISCLAIMER: Past performance does not guarantee future results."
echo "   This is for educational purposes only - NOT financial advice!"
