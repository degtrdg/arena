#!/bin/bash

# This script automates the process of training a steering vector, applying it to a model, evaluating the steered outputs, and visualizing the results.
# It should be run from the project root directory, e.g., `bash experiment1/experiment1.sh`.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- CONFIGURATION ---
# Get the absolute path to the directory where this script is located.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# The project root is one level up from the script directory.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Define the Python executable from the virtual environment using an absolute path.
# PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"

# Check if the Python executable exists. If not, try with '.exe'.
# if [ ! -f "$PYTHON_EXEC" ]; then
#     if [ -f "$PYTHON_EXEC.exe" ]; then
#         PYTHON_EXEC="$PYTHON_EXEC.exe"
#     else
#         echo "Error: Python executable not found in virtual environment."
#         echo "Looked for: $PROJECT_ROOT/venv/bin/python(.exe)"
#         exit 1
#     fi
# fi

# --- USER INPUTS ---
read -p "Enter the path to the initial dataset (e.g., experiment1/data/sycophancy_1000dp_cleaned.json): " INITIAL_FILE
if [ ! -f "$INITIAL_FILE" ]; then
    echo "Error: File not found at $INITIAL_FILE"
    exit 1
fi

read -sp "Enter your Gemini API key: " GEMINI_API_KEY
export GEMINI_API_KEY
echo # Add a newline for cleaner output

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: API key cannot be empty."
    exit 1
fi

# --- EXPERIMENT PARAMETERS (EDIT THESE) ---
TRAINING_MODE='specific'  # Options: 'specific', 'general'
# Used only if TRAINING_MODE is 'specific'.
SPECIFIC_CONTEXT_LABEL="technological_advancement"
MULTIPLIER=24.0
NUM_EXAMPLES=100
TARGET_LAYER=15
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"


# --- SCRIPT LOGIC ---
# Extract capability from the initial file path's name
FILENAME=$(basename "$INITIAL_FILE")
CAPABILITY=$(echo "$FILENAME" | cut -d'_' -f1)

echo "--- Starting Experiment Pipeline ---"
echo "Using Python from: $PYTHON_EXEC"
echo "Capability: $CAPABILITY"
echo "Initial File: $INITIAL_FILE"
echo "Training Mode: $TRAINING_MODE"
if [ "$TRAINING_MODE" = "specific" ]; then
    echo "Specific Context: $SPECIFIC_CONTEXT_LABEL"
fi
echo "-------------------------------------------"

# --- STEP 1: Train Steering Vector ---
echo "[1/4] Training steering vector..."
STEERING_VECTOR_PATH=""
while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == STEERING_VECTOR_PATH:* ]]; then
        STEERING_VECTOR_PATH=${line#STEERING_VECTOR_PATH:}
    fi
done < <("python" -u "experiment1/sv_with_bipo.py" \
    --training-mode "$TRAINING_MODE" \
    --capability "$CAPABILITY" \
    --specific-context-label "$SPECIFIC_CONTEXT_LABEL" \
    --dataset-path "$INITIAL_FILE" \
    --hf-token "$HF_TOKEN" \
    --model-name "$MODEL_NAME")

if [ -z "$STEERING_VECTOR_PATH" ]; then
    echo "Error: Could not find steering vector path in the output of sv_with_bipo.py."
    exit 1
fi
echo "Steering vector saved to: $STEERING_VECTOR_PATH"

# --- STEP 2: Steer Model with Vector ---
echo -e "\n[2/4] Applying steering vector to the model..."
STEERED_OUTPUT_PATH=""
while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == STEERED_OUTPUT_PATH:* ]]; then
        STEERED_OUTPUT_PATH=${line#STEERED_OUTPUT_PATH:}
    fi
done < <("python" -u "experiment1/steer.py" \
    --hf-token "$HF_TOKEN" \
    --model-name "$MODEL_NAME" \
    --sv-file "$STEERING_VECTOR_PATH" \
    --dataset-path "$INITIAL_FILE" \
    --target-layer "$TARGET_LAYER" \
    --multiplier "$MULTIPLIER" \
    --num-examples "$NUM_EXAMPLES")

if [ -z "$STEERED_OUTPUT_PATH" ]; then
    echo "Error: Could not find steered output path in the output of steer.py."
    exit 1
fi
echo "Steered outputs saved to: $STEERED_OUTPUT_PATH"

# --- STEP 3: Evaluate Steered Outputs ---
echo -e "\n[3/4] Evaluating steered outputs..."
EVAL_OUTPUT_FILENAME="eval_$(basename "$STEERED_OUTPUT_PATH")"
EVAL_OUTPUT_FILE="experiment1/eval_results/$EVAL_OUTPUT_FILENAME"
EVAL_OUTPUT_PATH=""

while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == EVAL_OUTPUT_PATH:* ]]; then
        EVAL_OUTPUT_PATH=${line#EVAL_OUTPUT_PATH:}
    fi
done < <("python" -u "experiment1/eval.py" \
    --capability "$CAPABILITY" \
    --input-file "$STEERED_OUTPUT_PATH" \
    --output-file "$EVAL_OUTPUT_FILE")

if [ -z "$EVAL_OUTPUT_PATH" ]; then
    echo "Error: Could not find eval output path in the output of eval.py."
    exit 1
fi
echo "Evaluation results saved to: $EVAL_OUTPUT_PATH"

# --- STEP 4: Visualize Results ---
echo -e "\n[4/4] Generating visualization..."
"python" -u "experiment1/vis.py" "$EVAL_OUTPUT_PATH"


echo -e "\n--- Pipeline Finished Successfully ---"
echo "Final visualization saved in the same directory as the evaluation results."