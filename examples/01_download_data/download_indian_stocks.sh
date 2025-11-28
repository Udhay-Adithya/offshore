#!/bin/bash
# =============================================================================
# Download Indian Stock Data (NSE & BSE)
# =============================================================================
# This script demonstrates how to download historical data for Indian stocks.
# 
# Ticker Format:
#   - NSE stocks: SYMBOL.NS (e.g., RELIANCE.NS, TCS.NS)
#   - BSE stocks: SYMBOL.BO (e.g., RELIANCE.BO, TCS.BO)
#   - NIFTY 50 Index: ^NSEI
#   - NIFTY Bank: ^NSEBANK
#   - Sensex: ^BSESN
#
# Usage: ./download_indian_stocks.sh
# =============================================================================

set -e  # Exit on error

echo "=================================================="
echo "🇮🇳 Downloading Indian Stock Data"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# =============================================================================
# LARGE CAP STOCKS (NSE)
# =============================================================================

echo ""
echo "📊 [LARGE CAP] Downloading Indian stocks from NSE..."
echo ""

# -----------------------------------------------------------------------------
# Reliance Industries - India's largest company by market cap
# -----------------------------------------------------------------------------
echo "  → RELIANCE.NS (Reliance Industries Ltd.)"
python -m src.cli.main download RELIANCE.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Tata Consultancy Services - IT Services Giant
# -----------------------------------------------------------------------------
echo ""
echo "  → TCS.NS (Tata Consultancy Services)"
python -m src.cli.main download TCS.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# HDFC Bank - Private Sector Bank
# -----------------------------------------------------------------------------
echo ""
echo "  → HDFCBANK.NS (HDFC Bank Ltd.)"
python -m src.cli.main download HDFCBANK.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Infosys - IT Services
# -----------------------------------------------------------------------------
echo ""
echo "  → INFY.NS (Infosys Ltd.)"
python -m src.cli.main download INFY.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# ICICI Bank - Private Sector Bank
# -----------------------------------------------------------------------------
echo ""
echo "  → ICICIBANK.NS (ICICI Bank Ltd.)"
python -m src.cli.main download ICICIBANK.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# =============================================================================
# BANKING & FINANCIAL STOCKS
# =============================================================================

echo ""
echo "📊 [BANKING] Downloading banking stocks..."
echo ""

# -----------------------------------------------------------------------------
# State Bank of India - Public Sector Bank
# -----------------------------------------------------------------------------
echo "  → SBIN.NS (State Bank of India)"
python -m src.cli.main download SBIN.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Kotak Mahindra Bank
# -----------------------------------------------------------------------------
echo ""
echo "  → KOTAKBANK.NS (Kotak Mahindra Bank)"
python -m src.cli.main download KOTAKBANK.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# =============================================================================
# IT SECTOR STOCKS
# =============================================================================

echo ""
echo "📊 [IT SECTOR] Downloading IT stocks..."
echo ""

# -----------------------------------------------------------------------------
# Wipro
# -----------------------------------------------------------------------------
echo "  → WIPRO.NS (Wipro Ltd.)"
python -m src.cli.main download WIPRO.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# HCL Technologies
# -----------------------------------------------------------------------------
echo ""
echo "  → HCLTECH.NS (HCL Technologies)"
python -m src.cli.main download HCLTECH.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Tech Mahindra
# -----------------------------------------------------------------------------
echo ""
echo "  → TECHM.NS (Tech Mahindra)"
python -m src.cli.main download TECHM.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# =============================================================================
# AUTO SECTOR
# =============================================================================

echo ""
echo "📊 [AUTO SECTOR] Downloading auto stocks..."
echo ""

# -----------------------------------------------------------------------------
# Tata Motors
# -----------------------------------------------------------------------------
echo "  → TATAMOTORS.NS (Tata Motors Ltd.)"
python -m src.cli.main download TATAMOTORS.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# -----------------------------------------------------------------------------
# Maruti Suzuki
# -----------------------------------------------------------------------------
echo ""
echo "  → MARUTI.NS (Maruti Suzuki India Ltd.)"
python -m src.cli.main download MARUTI.NS \
    --start 2018-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

# =============================================================================
# BSE EXAMPLES (Using .BO suffix)
# =============================================================================

echo ""
echo "📊 [BSE] Downloading stocks from BSE..."
echo ""

# -----------------------------------------------------------------------------
# Reliance on BSE
# -----------------------------------------------------------------------------
echo "  → RELIANCE.BO (Reliance Industries - BSE)"
python -m src.cli.main download RELIANCE.BO \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --interval 1d \
    --output data/raw

echo ""
echo "=================================================="
echo "✅ All Indian stocks downloaded successfully!"
echo "📁 Data saved to: data/raw/"
echo "=================================================="

# List downloaded files
echo ""
echo "📋 Downloaded Indian stock files:"
ls -la data/raw/*.NS*.csv data/raw/*.BO*.csv 2>/dev/null || echo "No Indian stock files found"
