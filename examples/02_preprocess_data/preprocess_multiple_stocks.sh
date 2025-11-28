#!/bin/bash
# =============================================================================
# Preprocess Multiple Stocks (Batch Processing)
# =============================================================================
# This script preprocesses multiple stocks in a batch.
# Useful for preparing data for comparative analysis or portfolio models.
#
# Usage: ./preprocess_multiple_stocks.sh
# =============================================================================

set -e  # Exit on error

echo "=================================================="
echo "🔧 Batch Preprocessing Multiple Stocks"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Define stocks to preprocess
# -----------------------------------------------------------------------------

# US Tech Stocks
US_STOCKS=("AAPL" "MSFT" "GOOGL" "NVDA" "TSLA")

# Indian Stocks (NSE)
INDIAN_STOCKS=("RELIANCE.NS" "TCS.NS" "HDFCBANK.NS" "INFY.NS" "ICICIBANK.NS")

# Market Indices
# INDICES=("^NSEI" "^GSPC")  # Uncomment to include indices

# -----------------------------------------------------------------------------
# Process US Stocks
# -----------------------------------------------------------------------------
echo ""
echo "🇺🇸 Processing US Stocks..."
echo ""

for TICKER in "${US_STOCKS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Processing: $TICKER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if raw data exists
    if [ ! -f "data/raw/${TICKER}_1d.csv" ]; then
        echo "⚠️  Raw data not found for $TICKER. Downloading first..."
        python -m src.cli.main download "$TICKER" \
            --start 2018-01-01 \
            --end 2024-01-01 \
            --interval 1d \
            --output data/raw
    fi
    
    # Preprocess
    python -m src.cli.main preprocess "$TICKER" \
        --config configs/data.yaml \
        --input data/raw \
        --output data/processed \
        --interval 1d
    
    echo ""
done

# -----------------------------------------------------------------------------
# Process Indian Stocks
# -----------------------------------------------------------------------------
echo ""
echo "🇮🇳 Processing Indian Stocks..."
echo ""

for TICKER in "${INDIAN_STOCKS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Processing: $TICKER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if raw data exists
    if [ ! -f "data/raw/${TICKER}_1d.csv" ]; then
        echo "⚠️  Raw data not found for $TICKER. Downloading first..."
        python -m src.cli.main download "$TICKER" \
            --start 2018-01-01 \
            --end 2024-01-01 \
            --interval 1d \
            --output data/raw
    fi
    
    # Preprocess
    python -m src.cli.main preprocess "$TICKER" \
        --config configs/data.yaml \
        --input data/raw \
        --output data/processed \
        --interval 1d
    
    echo ""
done

echo "=================================================="
echo "✅ Batch preprocessing complete!"
echo "=================================================="

# Summary
echo ""
echo "📋 Summary:"
echo "   US Stocks processed: ${#US_STOCKS[@]}"
echo "   Indian Stocks processed: ${#INDIAN_STOCKS[@]}"
echo ""
echo "📁 Processed data saved to: data/processed/"
echo ""

# List all processed files
echo "📋 Processed files:"
ls -la data/processed/*.csv 2>/dev/null | head -20 || echo "No processed files found"
