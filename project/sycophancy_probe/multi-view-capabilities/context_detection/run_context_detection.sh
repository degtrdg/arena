#!/bin/bash

# Test how different context-trained probes perform on a target dataset
# Usage: ./run_context_detection.sh [context_dataset_path] [vectors_folder] [layer_idx]

# Default parameters - modify as needed
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
CONTEXT_DATASET_PATH="${1:-data/refusal_100dp_1755665749.json}"

VECTORS_FOLDER="${2:-../experiment_pipeline/results/Qwen2.5-7B-Instruct/refusal}"
LAYER_IDX="${3:-16}"
OUTPUT_DIR="context_detection_results_refusal"

echo "Running context detection analysis..."
echo "Target dataset: $CONTEXT_DATASET_PATH"
echo "Probes folder: $VECTORS_FOLDER"
echo "Layer index: $LAYER_IDX"
echo "Output directory: $OUTPUT_DIR"
echo ""

python context_detection.py \
    --model_name "$MODEL_NAME" \
    --context_dataset_path "$CONTEXT_DATASET_PATH" \
    --vectors_folder "$VECTORS_FOLDER" \
    --layer_idx "$LAYER_IDX" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Analysis complete! Check the $OUTPUT_DIR directory for results."