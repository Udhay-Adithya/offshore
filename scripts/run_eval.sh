#!/bin/bash
# ==============================================================================
# Offshore: Evaluation Script
# ==============================================================================
# Usage: ./scripts/run_eval.sh --checkpoint PATH [OPTIONS]
#
# This script evaluates a trained model:
# 1. Loads the checkpoint
# 2. Runs evaluation metrics
# 3. Runs backtest
# 4. Generates plots
# ==============================================================================

set -e  # Exit on error

# Default values
CHECKPOINT=""
TICKER="AAPL"
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --ticker)
            TICKER="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --checkpoint PATH [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH     Path to model checkpoint (required)"
            echo "  --ticker TICKER       Stock ticker (default: AAPL)"
            echo "  --output-dir DIR      Output directory (default: checkpoint_dir/eval)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "${CHECKPOINT}" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# Set default output directory if not specified
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="$(dirname ${CHECKPOINT})/eval"
fi

echo "=============================================="
echo "Offshore Evaluation Pipeline"
echo "=============================================="
echo "Checkpoint:    ${CHECKPOINT}"
echo "Ticker:        ${TICKER}"
echo "Output Dir:    ${OUTPUT_DIR}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run evaluation
echo ""
echo "Running evaluation..."
python -m src.cli.main eval \
    --checkpoint "${CHECKPOINT}" \
    --data-config configs/data.yaml \
    --eval-config configs/eval.yaml \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
