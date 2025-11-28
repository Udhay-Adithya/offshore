#!/bin/bash
# =============================================================================
# Download US Stock Data
# =============================================================================
# This script demonstrates how to download historical data for US stocks.
# 
# Usage: ./download_us_stocks.sh
# =============================================================================

set -e  # Exit on error

echo "=================================================="
echo "📥 Downloading US Stock Data"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# -----------------------------------------------------------------------------
# Example 1: Download Apple (AAPL) - Tech Giant
# -----------------------------------------------------------------------------
echo ""
echo "📊 Downloading AAPL (Apple Inc.)..."
python -m src.cli.main download AAPL \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Example 2: Download Microsoft (MSFT) - Tech Giant
# -----------------------------------------------------------------------------
echo ""
echo "📊 Downloading MSFT (Microsoft Corporation)..."
python -m src.cli.main download MSFT \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Example 3: Download Google (GOOGL) - Tech Giant
# -----------------------------------------------------------------------------
echo ""
echo "📊 Downloading GOOGL (Alphabet Inc.)..."
python -m src.cli.main download GOOGL \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Example 4: Download Tesla (TSLA) - EV/Auto
# -----------------------------------------------------------------------------
echo ""
echo "📊 Downloading TSLA (Tesla Inc.)..."
python -m src.cli.main download TSLA \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Example 5: Download NVIDIA (NVDA) - Semiconductors
# -----------------------------------------------------------------------------
echo ""
echo "📊 Downloading NVDA (NVIDIA Corporation)..."
python -m src.cli.main download NVDA \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Example 6: Download Amazon (AMZN) - E-commerce/Cloud
# -----------------------------------------------------------------------------
echo ""
echo "📊 Downloading AMZN (Amazon.com Inc.)..."
python -m src.cli.main download AMZN \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

echo ""
echo "=================================================="
echo "✅ All US stocks downloaded successfully!"
echo "📁 Data saved to: data/raw/"
echo "=================================================="

# List downloaded files
echo ""
echo "📋 Downloaded files:"
ls -la data/raw/*.csv 2>/dev/null || echo "No files found"
