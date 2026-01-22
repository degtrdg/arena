#!/usr/bin/env python3

import json
import os
import shutil
import numpy as np
from collections import defaultdict

def load_results_from_directory(results_dir):
    """Load results from the directory structure"""
    all_context_results = {}
    
    context_dirs = [d for d in os.listdir(results_dir) 
                   if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith('.')]
    
    for context in context_dirs:
        context_path = os.path.join(results_dir, context)
        summary_file = os.path.join(context_path, 'all_layers_summary.json')
        
        if os.path.exists(summary_file):
            print(f"Loading results for context: {context}")
            
            with open(summary_file, 'r') as f:
                context_data = json.load(f)
            
            formatted_data = {}
            for layer_key, layer_data in context_data.items():
                if 'accuracy' in layer_data:
                    formatted_data[layer_key] = layer_data['accuracy']
            
            all_context_results[context] = formatted_data
        else:
            print(f"Warning: No summary file found for context {context}")
    
    return all_context_results

def find_important_layers(all_context_results):
    """
    Find important layers for each context:
    1. Top 2 layers with highest in-context performance per context
    2. Layer with best overall mean performance across all contexts
    """
    important_layers = {}
    all_layer_means = defaultdict(list)
    
    # Process each context
    for context, context_data in all_context_results.items():
            
        in_context_key = f"{context} (in-context)"
        layer_performances = []
        
        # Get in-context performance for each layer
        for layer_key, layer_data in context_data.items():
            if layer_key.startswith('layer_'):
                layer_num = int(layer_key.split('_')[1])
                if in_context_key in layer_data:
                    accuracy = layer_data[in_context_key]
                    layer_performances.append((layer_num, accuracy))
                    
                    # Also collect for overall mean calculation
                    # Calculate mean across all test contexts for this layer
                    all_accs = list(layer_data.values())
                    layer_mean = np.mean(all_accs)
                    all_layer_means[layer_num].append(layer_mean)
        
        # Sort by accuracy and get top 2
        layer_performances.sort(key=lambda x: x[1], reverse=True)
        top_2_layers = [layer_num for layer_num, _ in layer_performances[:2]]
        
        important_layers[context] = {
            'top_2': top_2_layers,
            'performances': layer_performances
        }
        
        print(f"{context}:")
        print(f"  Top 2 layers: {top_2_layers}")
        for layer_num, acc in layer_performances[:2]:
            print(f"    Layer {layer_num}: {acc:.3f}")
    
    # Find layer with best overall mean performance
    overall_layer_means = {}
    for layer_num, means in all_layer_means.items():
        overall_layer_means[layer_num] = np.mean(means)
    
    best_overall_layer = max(overall_layer_means.keys(), key=lambda k: overall_layer_means[k])
    best_overall_mean = overall_layer_means[best_overall_layer]
    
    print(f"\nBest overall layer: {best_overall_layer} (mean: {best_overall_mean:.3f})")
    
    return important_layers, best_overall_layer

def cleanup_directories(results_dir, important_layers, best_overall_layer, dry_run=False):
    """
    Remove layer directories that are not important, keeping only:
    1. Top 2 layers per context
    2. Best overall layer (for all contexts)
    3. all_layers_summary.json files
    """
    
    for context in important_layers.keys():
        context_path = os.path.join(results_dir, context)
        if not os.path.exists(context_path):
            continue
            
        # Get layers to keep for this context
        layers_to_keep = set(important_layers[context]['top_2'])
        layers_to_keep.add(best_overall_layer)  # Always keep the best overall layer
        
        print(f"\n{context}:")
        print(f"  Keeping layers: {sorted(layers_to_keep)}")
        
        # Get all layer directories
        layer_dirs = [d for d in os.listdir(context_path) 
                     if d.startswith('layer_') and os.path.isdir(os.path.join(context_path, d))]
        
        removed_count = 0
        for layer_dir in layer_dirs:
            layer_num = int(layer_dir.split('_')[1])
            layer_path = os.path.join(context_path, layer_dir)
            
            if layer_num not in layers_to_keep:
                if dry_run:
                    print(f"    Would remove: {layer_dir}")
                else:
                    shutil.rmtree(layer_path)
                    print(f"    Removed: {layer_dir}")
                removed_count += 1
        
        if not dry_run:
            print(f"  Removed {removed_count} layer directories")
        else:
            print(f"  Would remove {removed_count} layer directories")

def main():
    results_dir = "/Users/ishaagarwal/Documents/ERA/multi-view-capabilities/experiment_pipeline/results/linear_probe_results_refusal_all_layers"
    
    print("Loading results...")
    all_context_results = load_results_from_directory(results_dir)
    
    print("\nAnalyzing layer performance...")
    important_layers, best_overall_layer = find_important_layers(all_context_results)
    
    print(f"\n{'='*50}")
    print("CLEANUP SUMMARY")
    print(f"{'='*50}")
    
    # First do a dry run to show what would be removed
    print("\nDRY RUN - What would be removed:")
    cleanup_directories(results_dir, important_layers, best_overall_layer, dry_run=True)
    
    # Proceed with cleanup automatically
    print(f"\n{'='*50}")
    print("Proceeding with cleanup...")
    cleanup_directories(results_dir, important_layers, best_overall_layer, dry_run=False)
    print("\nCleanup completed!")

if __name__ == "__main__":
    main()