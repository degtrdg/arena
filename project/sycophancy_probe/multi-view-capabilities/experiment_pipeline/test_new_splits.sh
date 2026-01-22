#!/bin/bash

# Test script for the modified linear_probe.py with pre-split functionality
# This script demonstrates both usage modes: pre-existing splits and auto-generated splits

echo "Testing modified linear_probe.py with pre-split functionality..."
echo "============================================================"

# Test 1: Using existing pre-split data
echo "Test 1: Using existing pre-split data"
echo "--------------------------------------"
python3 linear_probe.py \
    --splits_dir truthfulness_splits_balanced \
    --target_layers 8 \
    --output_dir test_results_presplit \
    --model_name gpt2 \
    --num_epochs 5

echo ""
echo "Test 1 completed!"
echo ""

# Test 2: Auto-generating splits from dataset 
echo "Test 2: Auto-generating splits from dataset"
echo "--------------------------------------------"
python3 linear_probe.py \
    --dataset_path ../synth_data/data/truthfulness/truthfulness_1000dp_d_e.json \
    --target_layers 8 \
    --output_dir test_results_autosplit \
    --model_name gpt2 \
    --num_epochs 5

echo ""
echo "Test 2 completed!"
echo ""

# Test 3: Using existing auto-generated splits (should reuse from Test 2)
echo "Test 3: Reusing auto-generated splits"
echo "--------------------------------------"
python3 linear_probe.py \
    --dataset_path ../synth_data/data/truthfulness/truthfulness_1000dp_d_e.json \
    --target_layers 8 \
    --output_dir test_results_reuse \
    --model_name gpt2 \
    --num_epochs 5

echo ""
echo "Test 3 completed!"
echo ""

echo "============================================================"
echo "All tests completed! Check the output directories:"
echo "  - test_results_presplit/"
echo "  - test_results_autosplit/"  
echo "  - test_results_reuse/"
echo ""
echo "Auto-generated splits should be in:"
echo "  - truthfulness_1000dp_d_e_splits/"