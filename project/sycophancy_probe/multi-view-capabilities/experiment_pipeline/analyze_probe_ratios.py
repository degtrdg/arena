#!/usr/bin/env python3
"""
Complete analysis pipeline for probe accuracy ratios.

This script computes ratios of general linear probe accuracy to contextually trained 
linear probe accuracy across all capabilities and contexts for all models, then 
creates intuitive visualizations to show the results.

Usage: python analyze_probe_ratios.py
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


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
        print(f"  Processing capability: {capability}")
        
        # Load general data
        general_json_path = os.path.join(capability_path, 'general', 'all_layers_summary.json')
        if not os.path.exists(general_json_path):
            print(f"    Warning: No general data found for {capability}")
            continue
            
        with open(general_json_path, 'r') as f:
            general_data = json.load(f)
        
        # Find the best general layer
        best_general_layer, _ = find_best_layer(general_data, 'general (in-context)')
        if best_general_layer is None:
            print(f"    Warning: No general (in-context) data found for {capability}")
            continue
            
        print(f"    Best general layer: {best_general_layer}")
        
        # Get all context directories (excluding general)
        context_dirs = [d for d in os.listdir(capability_path) 
                       if os.path.isdir(os.path.join(capability_path, d)) 
                       and d not in ['general', 'cross_layer_analysis']]
        
        capability_results = {}
        
        for context in context_dirs:
            context_json_path = os.path.join(capability_path, context, 'all_layers_summary.json')
            if not os.path.exists(context_json_path):
                print(f"      Warning: No data found for context {context}")
                continue
                
            with open(context_json_path, 'r') as f:
                context_data = json.load(f)
            
            # Find the best layer for this context
            context_in_context_key = f"{context} (in-context)"
            best_context_layer, context_accuracy = find_best_layer(context_data, context_in_context_key)
            
            if best_context_layer is None or context_accuracy == 0:
                print(f"      Warning: No in-context data found for {context}")
                continue
            
            # Get general vector's accuracy on this context
            general_accuracy_on_context = get_general_accuracy_on_context(
                general_data, best_general_layer, context
            )
            
            if context_accuracy > 0:
                ratio = general_accuracy_on_context / context_accuracy
                capability_results[context] = ratio
                print(f"      {context}: {general_accuracy_on_context:.4f} / {context_accuracy:.4f} = {ratio:.4f}")
            else:
                print(f"      Warning: Zero context accuracy for {context}")
        
        if capability_results:
            results[capability] = capability_results
    
    return results


def compute_all_ratios(base_path: str = "results") -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute ratios for all models in the results directory.
    
    Args:
        base_path: Path to results directory
    
    Returns:
        Dictionary with model -> capability -> {context: ratio} mapping
    """
    all_model_data = {}
    
    # Check if results directory exists
    if not os.path.exists(base_path):
        print(f"Error: Results directory '{base_path}' does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for: {os.path.abspath(base_path)}")
        return {}
    
    # Get all model directories
    try:
        model_dirs = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))]
        print(f"Found model directories: {model_dirs}")
    except Exception as e:
        print(f"Error reading directory {base_path}: {e}")
        return {}
    
    for model_name in model_dirs:
        model_path = os.path.join(base_path, model_name)
        print(f"\n=== Processing model: {model_name} ===")
        
        # Process the model
        ratios = process_model(model_path)
        
        if ratios:
            all_model_data[model_name] = ratios
            
            # Save results to JSON file in the model directory
            output_file = os.path.join(model_path, f"{model_name}_probe_ratios.json")
            with open(output_file, 'w') as f:
                json.dump(ratios, f, indent=2)
            
            print(f"\n  Results saved to: {output_file}")
            
            # Print summary
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
    
    return all_model_data


def create_visualizations(model_data: Dict[str, Dict[str, Dict[str, float]]], output_dir: str):
    """Create simple, intuitive visualizations of the probe ratios - separate charts per model."""
    
    # Create separate charts for each model
    for model_name, data in model_data.items():
        # Prepare data for this model
        capabilities = []
        mean_ratios = []
        
        for capability, contexts in data.items():
            ratios = list(contexts.values())
            capabilities.append(capability.replace('_', ' ').title())
            mean_ratios.append(np.mean(ratios))
        
        # Sort by ratio value for better visualization
        sorted_pairs = sorted(zip(capabilities, mean_ratios), key=lambda x: x[1], reverse=True)
        capabilities, mean_ratios = zip(*sorted_pairs)
        
        # 1. Vertical bar chart for this model
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(range(len(capabilities)), mean_ratios, 
                       color='steelblue', alpha=0.7, edgecolor='navy')
        
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, 
                    label='Perfect Ratio (1.0)')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.8, 
                    label='95% Threshold')
        
        model_display_name = model_name.replace('-', ' ').replace('_', ' ')
        plt.title(f'{model_display_name}: Probe Accuracy Ratios by Capability\n(General vs Context-Specific)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Capabilities', fontsize=12, fontweight='bold')
        plt.ylabel('Average Ratio', fontsize=12, fontweight='bold')
        
        plt.xticks(range(len(capabilities)), capabilities, rotation=45, ha='right')
        plt.ylim(0.8, 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_capability_ratios.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Horizontal bar chart for this model (alternative view)
        plt.figure(figsize=(10, 8))
        
        # Sort ascending for horizontal layout
        sorted_pairs_asc = sorted(zip(capabilities, mean_ratios), key=lambda x: x[1], reverse=False)
        capabilities_h, mean_ratios_h = zip(*sorted_pairs_asc)
        
        bars = plt.barh(range(len(capabilities_h)), mean_ratios_h, 
                        color='lightcoral', alpha=0.7, edgecolor='darkred')
        
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='Perfect Ratio (1.0)')
        plt.axvline(x=0.95, color='orange', linestyle='--', alpha=0.8, label='95% Threshold')
        
        plt.title(f'{model_display_name}: Capability Performance Ranking\n(Higher = General performs better relative to specialized)', 
                  fontsize=12, fontweight='bold', pad=20)
        plt.xlabel('Average Ratio', fontsize=11, fontweight='bold')
        plt.ylabel('Capabilities', fontsize=11, fontweight='bold')
        
        plt.yticks(range(len(capabilities_h)), capabilities_h)
        plt.xlim(0.85, 1.05)
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.002, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_capability_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. If multiple models, create a comparison chart
    if len(model_data) > 1:
        # Prepare data for comparison
        capabilities = []
        mean_ratios = []
        models = []
        
        for model_name, data in model_data.items():
            for capability, contexts in data.items():
                ratios = list(contexts.values())
                capabilities.append(capability.replace('_', ' ').title())
                mean_ratios.append(np.mean(ratios))
                models.append(model_name.replace('-', ' '))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Capability': capabilities,
            'Mean_Ratio': mean_ratios,
            'Model': models
        })
        
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        capabilities_unique = sorted(df['Capability'].unique())
        models_unique = sorted(df['Model'].unique())
        
        x = np.arange(len(capabilities_unique))
        width = 0.35 if len(models_unique) == 2 else 0.25
        
        colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, model in enumerate(models_unique):
            model_data_subset = df[df['Model'] == model]
            ratios = [model_data_subset[model_data_subset['Capability'] == cap]['Mean_Ratio'].iloc[0] 
                     if cap in model_data_subset['Capability'].values else 0 
                     for cap in capabilities_unique]
            
            bars = plt.bar(x + i * width, ratios, width, label=model, 
                          alpha=0.8, color=colors[i % len(colors)])
            
            # Add value labels
            for bar, ratio in zip(bars, ratios):
                if ratio > 0:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, label='Perfect Ratio')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.8, label='95% Threshold')
        
        plt.title('Average General Probe Performance to Contextual Probe Performance Ratio by Capability', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Capabilities', fontsize=12, fontweight='bold')
        plt.ylabel('Average General: Contextual Accuracy Ratio', fontsize=12, fontweight='bold')
        
        plt.xticks(x + width * (len(models_unique) - 1) / 2, capabilities_unique, rotation=45, ha='right')
        plt.ylim(0.8, 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_statistics(model_data: Dict[str, Dict[str, Dict[str, float]]], output_dir: str):
    """Create summary statistics table."""
    summary_data = []
    
    for model_name, data in model_data.items():
        for capability, contexts in data.items():
            ratios = list(contexts.values())
            summary_data.append({
                'Model': model_name,
                'Capability': capability.replace('_', ' ').title(),
                'Contexts': len(ratios),
                'Mean Ratio': np.mean(ratios),
                'Std Ratio': np.std(ratios),
                'Min Ratio': np.min(ratios),
                'Max Ratio': np.max(ratios),
                'Ratios > 1.0': sum(1 for r in ratios if r > 1.0),
                'Ratios < 0.9': sum(1 for r in ratios if r < 0.9)
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    
    print("\nSummary Statistics:")
    print(df.round(4).to_string(index=False))


def main():
    """Main function to run the complete analysis pipeline."""
    print("=== Probe Ratio Analysis Pipeline ===")
    print("1. Computing ratios from raw data...")
    
    # Compute all ratios from the results
    model_data = compute_all_ratios()
    
    if not model_data:
        print("No model data found! Make sure you have results in the 'results' directory.")
        return
    
    print(f"\n2. Creating visualizations for models: {list(model_data.keys())}")
    
    # Create output directory
    output_dir = "ratio_visualizations"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create visualizations
    create_visualizations(model_data, output_dir)
    print("  ✓ Simple bar charts created")
    
    # Create summary statistics
    print("\n3. Generating summary statistics...")
    create_summary_statistics(model_data, output_dir)
    print("  ✓ Summary statistics saved")
    
    print(f"\n=== Analysis Complete! ===")
    print(f"All results saved to: {output_dir}/")
    print("\nKey files created:")
    
    # List individual model files
    for model_name in model_data.keys():
        print(f"  - {model_name}_capability_ratios.png: Vertical bar chart for {model_name}")
        print(f"  - {model_name}_capability_ranking.png: Horizontal ranking for {model_name}")
    
    if len(model_data) > 1:
        print(f"  - model_comparison.png: Side-by-side comparison across models")
    print("  - summary_statistics.csv: Detailed statistics table")
    print("\nJSON files with detailed ratios saved to each model's results directory.")


if __name__ == "__main__":
    main()