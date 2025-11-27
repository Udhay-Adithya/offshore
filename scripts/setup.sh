#!/bin/bash
# ==============================================================================
# Offshore: Setup Script
# ==============================================================================
# Usage: ./scripts/setup.sh
#
# This script sets up the development environment:
# 1. Creates virtual environment
# 2. Installs dependencies
# 3. Runs tests to verify setup
# ==============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Offshore Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: ${PYTHON_VERSION}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode
echo ""
echo "Installing offshore in editable mode..."
pip install -e .

# Install dev dependencies
echo ""
echo "Installing dev dependencies..."
pip install pytest pytest-cov black isort mypy

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw data/processed outputs

# Run tests
echo ""
echo "Running tests to verify setup..."
python -m pytest tests/ -v --tb=short || true

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the CLI:"
echo "  python -m src.cli.main --help"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo "=============================================="
