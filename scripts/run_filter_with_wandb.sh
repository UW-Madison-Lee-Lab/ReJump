#!/bin/bash

# Script to run filter_correct_responses.py with wandb logging
# Usage: ./run_filter_with_wandb.sh input_path output_path already_trained_path [run_name]

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 input_path output_path already_trained_path [run_name]"
    echo "Example: $0 data/responses.parquet data/filtered.parquet data/correct_responses.parquet filter-run-1"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
ALREADY_TRAINED_PATH=$3
RUN_NAME=${4:-"filter-responses-$(date +%Y%m%d-%H%M%S)"}

echo "Running filter with wandb logging..."
echo "Input path: $INPUT_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Already trained path: $ALREADY_TRAINED_PATH"
echo "Run name: $RUN_NAME"

python filter_correct_responses.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --already_trained_correct_path "$ALREADY_TRAINED_PATH" \
    --wandb_run_name "$RUN_NAME"

echo "Filtering completed!" 