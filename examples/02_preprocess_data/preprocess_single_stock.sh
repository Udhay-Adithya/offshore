#!/bin/bash
# =============================================================================
# Preprocess Single Stock Data
# =============================================================================
# This script preprocesses downloaded stock data for a single ticker.
# It cleans data, adds technical indicators, creates labels, and splits
# into train/val/test sets.
#
# Usage: ./preprocess_single_stock.sh [TICKER]
# Example: ./preprocess_single_stock.sh AAPL
#          ./preprocess_single_stock.sh RELIANCE.NS
# =============================================================================

set -e  # Exit on error

# Get ticker from argument or use default
TICKER=${1:-AAPL}

echo "=================================================="
echo "🔧 Preprocessing Data for: $TICKER"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Basic Preprocessing with Default Config
# -----------------------------------------------------------------------------
echo ""
echo "📋 Using configuration: configs/data.yaml"
echo ""

python -m src.cli.main preprocess "$TICKER" \
    --config configs/data.yaml \
    --input data/raw \
    --output data/processed \
    --interval 1d

echo ""
echo "=================================================="
echo "✅ Preprocessing complete for $TICKER"
echo "=================================================="

# Show output files
echo ""
echo "📁 Output files:"
ls -la data/processed/${TICKER}* 2>/dev/null || echo "Files saved with different naming convention"

echo ""
echo "💡 Next step: Train a model with:"
echo "   python -m src.cli.main train --ticker $TICKER"
