#!/usr/bin/env python3
"""
Compute ratios of general linear probe accuracy to contextually trained linear probe accuracy
across all capabilities and contexts for a given model.

This script processes the experiment results for each capability (folder) and computes
the ratio of general vector accuracy on each context divided by the context vector's 
accuracy on the same context.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any


def find_best_layer(all_layers_data: Dict[str, Any], target_context: str) -> Tuple[str, float]:
    """
    Find the layer with the highest accuracy for the target context.
    
    Args:
        all_layers_data: Dictionary containing all layer data
        target_context: The context to find the best accuracy for (e.g., "general (in-context)")
    
    Returns:
        Tuple of (best_layer_name, best_accuracy)
    """
    best_layer = None
    best_accuracy = -1.0
    
    for layer_name, layer_data in all_layers_data.items():
        if 'accuracy' in layer_data and target_context in layer_data['accuracy']:
            accuracy = layer_data['accuracy'][target_context]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer_name
    
    return best_layer, best_accuracy


def get_general_accuracy_on_context(general_data: Dict[str, Any], best_general_layer: str, context: str) -> float:
    """
    Get the general vector's accuracy on a specific context.
    
    Args:
        general_data: General folder's all_layers_summary.json data
        best_general_layer: The layer with the best general performance
        context: The context to get accuracy for
    
    Returns:
        Accuracy value
    """
    if best_general_layer in general_data and 'accuracy' in general_data[best_general_layer]:
        return general_data[best_general_layer]['accuracy'].get(context, 0.0)
    return 0.0


def process_model(model_path: str) -> Dict[str, Dict[str, float]]:
    """
    Process all capabilities for a single model and compute ratios.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        Dictionary with capability -> {context: ratio} mapping
    """
    results = {}
    
    # Get all capability directories
    capability_dirs = [d for d in os.listdir(model_path) 
                      if os.path.isdir(os.path.join(model_path, d)) 
                      and d != 'cross_layer_analysis']
    
    for capability in capability_dirs:
        capability_path = os.path.join(model_path, capability)
        print(f"Processing capability: {capability}")
        
        # Load general data
        general_json_path = os.path.join(capability_path, 'general', 'all_layers_summary.json')
        if not os.path.exists(general_json_path):
            print(f"  Warning: No general data found for {capability}")
            continue
            
        with open(general_json_path, 'r') as f:
            general_data = json.load(f)
        
        # Find the best general layer
        best_general_layer, _ = find_best_layer(general_data, 'general (in-context)')
        if best_general_layer is None:
            print(f"  Warning: No general (in-context) data found for {capability}")
            continue
            
        print(f"  Best general layer: {best_general_layer}")
        
        # Get all context directories (excluding general)
        context_dirs = [d for d in os.listdir(capability_path) 
                       if os.path.isdir(os.path.join(capability_path, d)) 
                       and d not in ['general', 'cross_layer_analysis']]
        
        capability_results = {}
        
        for context in context_dirs:
            context_json_path = os.path.join(capability_path, context, 'all_layers_summary.json')
            if not os.path.exists(context_json_path):
                print(f"    Warning: No data found for context {context}")
                continue
                
            with open(context_json_path, 'r') as f:
                context_data = json.load(f)
            
            # Find the best layer for this context
            context_in_context_key = f"{context} (in-context)"
            best_context_layer, context_accuracy = find_best_layer(context_data, context_in_context_key)
            
            if best_context_layer is None or context_accuracy == 0:
                print(f"    Warning: No in-context data found for {context}")
                continue
            
            # Get general vector's accuracy on this context
            general_accuracy_on_context = get_general_accuracy_on_context(
                general_data, best_general_layer, context
            )
            
            if context_accuracy > 0:
                ratio = general_accuracy_on_context / context_accuracy
                capability_results[context] = ratio
                print(f"    {context}: {general_accuracy_on_context:.4f} / {context_accuracy:.4f} = {ratio:.4f}")
            else:
                print(f"    Warning: Zero context accuracy for {context}")
        
        if capability_results:
            results[capability] = capability_results
    
    return results


def main():
    # Base results directory - using relative path from experiment_pipeline
    base_path = "results"
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d))]
    
    for model_name in model_dirs:
        model_path = os.path.join(base_path, model_name)
        print(f"\n=== Processing model: {model_name} ===")
        
        # Process the model
        ratios = process_model(model_path)
        
        # Save results to JSON file in the model directory
        output_file = os.path.join(model_path, f"{model_name}_probe_ratios.json")
        with open(output_file, 'w') as f:
            json.dump(ratios, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        print(f"\nSummary for {model_name}:")
        total_ratios = 0
        count = 0
        for capability, contexts in ratios.items():
            print(f"  {capability}: {len(contexts)} contexts")
            for context, ratio in contexts.items():
                total_ratios += ratio
                count += 1
        
        if count > 0:
            avg_ratio = total_ratios / count
            print(f"  Average ratio across all contexts: {avg_ratio:.4f}")
        print(f"  Total contexts processed: {count}")


if __name__ == "__main__":
    main()