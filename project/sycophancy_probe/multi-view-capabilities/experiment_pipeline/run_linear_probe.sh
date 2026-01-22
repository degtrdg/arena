#!/bin/bash

# Arguments
MODEL_NAME="mistralai/Mistral-7B-v0.1"
# CURRENTLY BASE IS SET TO TRUE

# DATASET_PATH="../synth_data/data/truthfulness_1000dp_d_e.json"
# DATASET_PATH="../synth_data/data/sycophancy_1000dp_cleaned_ver2_deduped.json"

DATASET_PATH="../synth_data/data/refusal/refusal_1000dp_d_e.json"
OUTPUT_PATH="results/Mistral-7b-v0.1/refusal"
# Run linear probes on ALL layers (default behavior)
python3 linear_probe.py \
    --model_name "$MODEL_NAME" \
    --target_layers 5 10 15 20 25 30 \
    --output_dir "$OUTPUT_PATH" \
    --splits_dir "../synth_data/data/refusal/refusal_1000dp_d_e_splits" \
    # --dataset_path "$DATASET_PATH" \

    # --all_layers

# Alternative: Run on specific layers (uncomment to use)
# echo "Running linear probes on specific layers..."
# python3 linear_probe.py \
#     --model_name "$MODEL_NAME" \
#     --dataset_path "$DATASET_PATH" \
#     --target_layers 20 30 40

# Alternative: Run on single layer (uncomment to use)  
# echo "Running linear probe on single layer..."
# python3 linear_probe.py \
#     --model_name "$MODEL_NAME" \
#     --dataset_path "$DATASET_PATH" \
#     --target_layers 40
