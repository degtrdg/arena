import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any

def find_best_layer_for_context(context_data: Dict[str, Dict], context_key: str) -> Tuple[str, float]:
    """Find the layer where the context has the highest in-context accuracy."""
    best_layer = None
    best_accuracy = -1
    
    for layer, layer_data in context_data.items():
        if "accuracy" in layer_data and context_key in layer_data["accuracy"]:
            accuracy = layer_data["accuracy"][context_key]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer
    
    return best_layer, best_accuracy

def calculate_generalization_ratio(context_data: Dict[str, Dict], context_key: str, 
                                 best_layer: str, other_contexts: List[str]) -> float:
    """Calculate generalization ratio: average accuracy on other contexts / in-context accuracy."""
    if best_layer not in context_data or "accuracy" not in context_data[best_layer]:
        return None
    
    layer_accuracies = context_data[best_layer]["accuracy"]
    in_context_accuracy = layer_accuracies.get(context_key, 0)
    
    if in_context_accuracy == 0:
        return None
    
    # Get accuracies on other contexts for the same layer
    other_accuracies = []
    for other_context in other_contexts:
        if other_context in layer_accuracies:
            other_accuracies.append(layer_accuracies[other_context])
    
    if not other_accuracies:
        return None
    
    avg_other_accuracy = np.mean(other_accuracies)
    generalization_ratio = avg_other_accuracy / in_context_accuracy
    
    return generalization_ratio

def process_capability_folder(capability_path: str, capability_name: str) -> Dict[str, Any]:
    """Process a single capability folder to calculate generalization ratios."""
    results = {
        "capability": capability_name,
        "contexts": {}
    }
    
    # Get all context folders (exclude cross_layer_analysis and general)
    context_folders = []
    for item in os.listdir(capability_path):
        item_path = os.path.join(capability_path, item)
        if os.path.isdir(item_path) and item not in ["cross_layer_analysis", "general"]:
            context_folders.append(item)
    
    print(f"Processing capability: {capability_name}")
    print(f"Found contexts: {context_folders}")
    
    # Load data for all contexts
    context_data = {}
    for context_folder in context_folders:
        summary_file = os.path.join(capability_path, context_folder, "all_layers_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                context_data[context_folder] = json.load(f)
    
    # Calculate generalization ratios for each context
    for context_a in context_folders:
        if context_a not in context_data:
            continue
            
        # Find the in-context key (usually has "(in-context)" suffix)
        context_key = None
        for layer, layer_data in context_data[context_a].items():
            if "accuracy" in layer_data:
                for key in layer_data["accuracy"].keys():
                    if "(in-context)" in key or context_a.replace("_", " ") in key:
                        context_key = key
                        break
                if context_key:
                    break
        
        if not context_key:
            print(f"Warning: Could not find in-context key for {context_a}")
            continue
        
        print(f"  Processing context: {context_a} with key: {context_key}")
        
        # Find best layer for this context
        best_layer, best_accuracy = find_best_layer_for_context(context_data[context_a], context_key)
        
        if not best_layer:
            print(f"    Warning: Could not find best layer for {context_a}")
            continue
        
        print(f"    Best layer: {best_layer}, Best accuracy: {best_accuracy:.4f}")
        
        # Get other contexts' keys from the same layer
        other_context_keys = []
        if best_layer in context_data[context_a] and "accuracy" in context_data[context_a][best_layer]:
            layer_accuracies = context_data[context_a][best_layer]["accuracy"]
            for key in layer_accuracies.keys():
                if key != context_key and key != "general":
                    other_context_keys.append(key)
        
        # Calculate generalization ratio
        generalization_ratio = calculate_generalization_ratio(
            context_data[context_a], context_key, best_layer, other_context_keys
        )
        
        if generalization_ratio is not None:
            print(f"    Generalization ratio: {generalization_ratio:.4f}")
            results["contexts"][context_a] = {
                "best_layer": best_layer,
                "in_context_accuracy": best_accuracy,
                "generalization_ratio": generalization_ratio,
                "context_key": context_key,
                "other_contexts_evaluated": other_context_keys
            }
        else:
            print(f"    Warning: Could not calculate generalization ratio for {context_a}")
    
    return results

def create_generalization_ratio_graph(results: Dict[str, Any], model_name: str, output_dir: str):
    """Create bar graph of generalization ratios for each capability."""
    capabilities = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, capability in enumerate(capabilities):
        if i >= len(axes):
            break
            
        capability_data = results[capability]
        contexts = list(capability_data["contexts"].keys())
        ratios = [capability_data["contexts"][ctx]["generalization_ratio"] for ctx in contexts]
        
        ax = axes[i]
        bars = ax.bar(range(len(contexts)), ratios, alpha=0.8)
        ax.set_title(f'{capability.title()} - Generalization Ratios')
        ax.set_xlabel('Contexts')
        ax.set_ylabel('Generalization Ratio (Other/Self)')
        ax.set_xticks(range(len(contexts)))
        ax.set_xticklabels([ctx.replace('_', ' ').title() for ctx in contexts], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', 
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", 
                       ha='center', va='bottom', fontsize=8)
        
        # Add horizontal line at y=1 (perfect generalization)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Generalization')
        ax.legend()
    
    # Hide unused subplots
    for i in range(len(capabilities), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"generalization_ratios_{model_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generalization ratio graph saved to: {output_path}")

def process_model_results(model_path: str, model_name: str) -> Dict[str, Any]:
    """Process all capabilities for a single model."""
    results = {}
    
    # Get all capability folders
    capability_folders = []
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path) and not item.endswith('.json'):
            capability_folders.append(item)
    
    print(f"\nProcessing model: {model_name}")
    print(f"Found capabilities: {capability_folders}")
    
    for capability in capability_folders:
        capability_path = os.path.join(model_path, capability)
        capability_results = process_capability_folder(capability_path, capability)
        results[capability] = capability_results
    
    return results

def main():
    """Main function to process all models and calculate generalization ratios."""
    base_dir = "/Users/ishaagarwal/Documents/ERA/multi-view-capabilities/experiment_pipeline/results"
    
    # Find all model directories
    model_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            model_dirs.append(item)
    
    print(f"Found {len(model_dirs)} model directories: {model_dirs}")
    
    all_results = {}
    
    for model_name in model_dirs:
        model_path = os.path.join(base_dir, model_name)
        try:
            model_results = process_model_results(model_path, model_name)
            all_results[model_name] = model_results
            
            # Save individual model results
            output_file = os.path.join(base_dir, f"generalization_ratios_{model_name}.json")
            with open(output_file, 'w') as f:
                json.dump(model_results, f, indent=2)
            print(f"Results saved to: {output_file}")
            
            # Create visualization
            create_generalization_ratio_graph(model_results, model_name, base_dir)
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue
    
    # Save combined results
    combined_output = os.path.join(base_dir, "all_generalization_ratios.json")
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll generalization ratio results saved to: {combined_output}")
    print(f"Generated visualization graphs for {len(all_results)} models")

if __name__ == "__main__":
    main()