#!/bin/bash
# =============================================================================
# Download Market Indices
# =============================================================================
# This script demonstrates how to download major market indices.
#
# Index Tickers:
#   - ^NSEI    : NIFTY 50 (NSE India)
#   - ^NSEBANK : NIFTY Bank (Banking sector)
#   - ^BSESN   : Sensex (BSE India)
#   - ^GSPC    : S&P 500 (US)
#   - ^DJI     : Dow Jones Industrial Average (US)
#   - ^IXIC    : NASDAQ Composite (US)
#   - ^FTSE    : FTSE 100 (UK)
#   - ^N225    : Nikkei 225 (Japan)
#
# Usage: ./download_indices.sh
# =============================================================================

set -e  # Exit on error

echo "=================================================="
echo "📈 Downloading Market Indices"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# =============================================================================
# INDIAN INDICES
# =============================================================================

echo ""
echo "🇮🇳 [INDIA] Downloading Indian market indices..."
echo ""

# -----------------------------------------------------------------------------
# NIFTY 50 - Top 50 NSE stocks
# -----------------------------------------------------------------------------
echo "  → ^NSEI (NIFTY 50 Index)"
python -m src.cli.main download "^NSEI" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# NIFTY Bank - Banking sector index
# -----------------------------------------------------------------------------
echo ""
echo "  → ^NSEBANK (NIFTY Bank Index)"
python -m src.cli.main download "^NSEBANK" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Sensex - BSE 30 stocks
# -----------------------------------------------------------------------------
echo ""
echo "  → ^BSESN (BSE Sensex)"
python -m src.cli.main download "^BSESN" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# =============================================================================
# US INDICES
# =============================================================================

echo ""
echo "🇺🇸 [USA] Downloading US market indices..."
echo ""

# -----------------------------------------------------------------------------
# S&P 500
# -----------------------------------------------------------------------------
echo "  → ^GSPC (S&P 500 Index)"
python -m src.cli.main download "^GSPC" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Dow Jones Industrial Average
# -----------------------------------------------------------------------------
echo ""
echo "  → ^DJI (Dow Jones Industrial Average)"
python -m src.cli.main download "^DJI" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# NASDAQ Composite
# -----------------------------------------------------------------------------
echo ""
echo "  → ^IXIC (NASDAQ Composite)"
python -m src.cli.main download "^IXIC" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# =============================================================================
# GLOBAL INDICES
# =============================================================================

echo ""
echo "🌍 [GLOBAL] Downloading global indices..."
echo ""

# -----------------------------------------------------------------------------
# FTSE 100 (UK)
# -----------------------------------------------------------------------------
echo "  → ^FTSE (FTSE 100 - UK)"
python -m src.cli.main download "^FTSE" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Nikkei 225 (Japan)
# -----------------------------------------------------------------------------
echo ""
echo "  → ^N225 (Nikkei 225 - Japan)"
python -m src.cli.main download "^N225" \
    --start 2015-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

echo ""
echo "=================================================="
echo "✅ All indices downloaded successfully!"
echo "📁 Data saved to: data/raw/"
echo "=================================================="

# List downloaded files
echo ""
echo "📋 Downloaded index files:"
ls -la data/raw/^*.csv 2>/dev/null || echo "No index files found (files may have different naming)"
