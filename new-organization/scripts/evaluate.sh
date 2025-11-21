#!/bin/bash
# Evaluation script for stock prediction models
# Usage: ./scripts/evaluate.sh [model_name] [model_type] [time_args]

# Activate conda environment if available
if command -v conda &> /dev/null; then
    conda activate csci566-project 2>/dev/null || true
fi

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
MODEL_NAME="${1:-savedmodel_classification_cnn_lstm.pth}"
MODEL_TYPE="${2:-cnn_lstm}"
TIME_ARGS="${3:-1990-01-01,2015-12-31}"

echo "=========================================="
echo "Model Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Type: $MODEL_TYPE"
echo "Time Args: $TIME_ARGS"
echo "=========================================="

# Run evaluation
python evaluation/evaluate_model.py "$MODEL_NAME" "$MODEL_TYPE" "$TIME_ARGS"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    # Extract model name without extension for results file naming
    MODEL_BASE=$(basename "$MODEL_NAME" .pth)
    RESULTS_FILE="evaluation_results_${MODEL_BASE}.json"
    
    # Run evaluation analysis if results file exists
    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo "=========================================="
        echo "Running Evaluation Analysis"
        echo "=========================================="
        
        ANALYSIS_OUTPUT="results/${MODEL_TYPE}/evaluation_analysis_report_${MODEL_BASE}.json"
        PORTFOLIO_CSV="results/${MODEL_TYPE}/portfolio_table_${MODEL_BASE}.csv"
        
        # Create results directory if it doesn't exist
        mkdir -p "results/${MODEL_TYPE}"
        
        python evaluation/evaluation_analysis.py \
            --input "$RESULTS_FILE" \
            --output "$ANALYSIS_OUTPUT" \
            --portfolio-table-csv "$PORTFOLIO_CSV"
        
        echo ""
        echo "=========================================="
        echo "Evaluation Complete!"
        echo "=========================================="
        echo "Results: $RESULTS_FILE"
        echo "Analysis: $ANALYSIS_OUTPUT"
        echo "Portfolio Table: $PORTFOLIO_CSV"
        echo ""
        echo "To view TensorBoard logs:"
        echo "  tensorboard --logdir runs/evaluation"
    fi
else
    echo ""
    echo "=========================================="
    echo "Evaluation Failed!"
    echo "=========================================="
    exit 1
fi

