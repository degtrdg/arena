import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any

def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """Extract capability and model from folder name."""
    pattern = r"context_detection_results_(.+)_(.+)"
    match = re.match(pattern, folder_name)
    if match:
        capability = match.group(1)
        model = match.group(2)
        return capability, model
    else:
        raise ValueError(f"Cannot parse folder name: {folder_name}")

def extract_best_layer_data(context_folder_path: str, context_name: str) -> Dict[str, Any]:
    """Find the layer with highest accuracy for a specific context."""
    best_score = -1
    best_layer_data = None
    
    for filename in os.listdir(context_folder_path):
        if filename.startswith("context_detection_summary_layer_") and filename.endswith(".json"):
            file_path = os.path.join(context_folder_path, filename)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            context_scores = data["context_scores"]
            
            # Find context score that matches the context name
            context_score = None
            for score_key, score_value in context_scores.items():
                if context_name.replace(" ", "_").replace("-", "_") in score_key or score_key.replace("_", " ") == context_name:
                    context_score = score_value
                    break
            
            # If exact match not found, try partial matching
            if context_score is None:
                for score_key, score_value in context_scores.items():
                    if score_key != "general" and context_name.lower() in score_key.lower():
                        context_score = score_value
                        break
            
            if context_score is not None and context_score > best_score:
                best_score = context_score
                general_score = context_scores.get("general", 0)
                best_layer_data = {
                    "layer": data["layer_idx"],
                    "context_score": context_score,
                    "general_score": general_score,
                    "context_key": score_key if 'score_key' in locals() else None
                }
    
    return best_layer_data

def create_bar_graph(results_data: Dict[str, Dict], capability: str, model: str, output_dir: str):
    """Create bar graph comparing context scores vs general scores."""
    contexts = list(results_data.keys())
    context_scores = [results_data[ctx]["context_score"] for ctx in contexts]
    general_scores = [results_data[ctx]["general_score"] for ctx in contexts]
    
    x = np.arange(len(contexts))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, context_scores, width, label='Context Score', alpha=0.8)
    bars2 = ax.bar(x + width/2, general_scores, width, label='General Score', alpha=0.8)
    
    ax.set_xlabel('Contexts')
    ax.set_ylabel('Context Detection Score')
    ax.set_title(f'Context vs General Probe Context Detection for {capability} - {model}')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"bar_graph_{capability}_{model}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bar graph saved to: {output_path}")

def process_capability_model_folder(folder_path: str, base_dir: str) -> Dict[str, Dict]:
    """Process a single capability-model folder."""
    folder_name = os.path.basename(folder_path)
    capability, model = parse_folder_name(folder_name)
    
    results = {}
    
    # Get all subdirectories (contexts)
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            context_name = item
            print(f"Processing context: {context_name} for {capability}-{model}")
            
            best_data = extract_best_layer_data(item_path, context_name)
            if best_data:
                results[context_name] = best_data
            else:
                print(f"Warning: No data found for context {context_name}")
    
    # Save results to JSON
    output_file = os.path.join(base_dir, f"best_layers_{capability}_{model}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Create bar graph
    if results:
        create_bar_graph(results, capability, model, base_dir)
    
    return results

def main():
    """Main function to process all context detection results."""
    base_dir = "/Users/ishaagarwal/Documents/ERA/multi-view-capabilities/context_detection"
    
    # Find all context_detection_results folders
    result_folders = []
    for item in os.listdir(base_dir):
        if item.startswith("context_detection_results_") and "_" in item[len("context_detection_results_"):]:
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                result_folders.append(item_path)
    
    print(f"Found {len(result_folders)} result folders to process")
    
    all_results = {}
    
    for folder_path in result_folders:
        try:
            folder_name = os.path.basename(folder_path)
            print(f"\nProcessing folder: {folder_name}")
            
            results = process_capability_model_folder(folder_path, base_dir)
            all_results[folder_name] = results
            
        except Exception as e:
            print(f"Error processing {folder_path}: {e}")
            continue
    
    # Save combined results
    combined_output = os.path.join(base_dir, "all_best_layers_results.json")
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to: {combined_output}")
    print(f"Generated {len(all_results)} bar graphs")

if __name__ == "__main__":
    main()