import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

def load_generalization_data(file_path: str) -> Dict:
    """Load generalization ratio data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_capability_averages(model_data: Dict) -> Dict[str, float]:
    """Calculate average generalization ratio for each capability."""
    capability_averages = {}
    
    for capability, capability_data in model_data.items():
        if 'contexts' in capability_data:
            ratios = []
            for context, context_data in capability_data['contexts'].items():
                if 'generalization_ratio' in context_data:
                    ratios.append(context_data['generalization_ratio'])
            
            if ratios:
                capability_averages[capability] = np.mean(ratios)
    
    return capability_averages

def create_aggregate_comparison(all_data: Dict, output_dir: str):
    """Create simple comparison plots across capabilities and models."""
    
    # Calculate averages for each model
    model_averages = {}
    for model_name, model_data in all_data.items():
        model_averages[model_name] = calculate_capability_averages(model_data)
    
    # Get all capabilities
    capabilities = set()
    for model_data in model_averages.values():
        capabilities.update(model_data.keys())
    capabilities = sorted(list(capabilities))
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average generalization ratio by capability
    cap_means = []
    cap_stds = []
    cap_labels = []
    
    for cap in capabilities:
        values = []
        for model_data in model_averages.values():
            if cap in model_data:
                values.append(model_data[cap])
        
        if values:
            cap_means.append(np.mean(values))
            cap_stds.append(np.std(values) if len(values) > 1 else 0)
            cap_labels.append(cap.title())
    
    bars1 = ax1.bar(range(len(cap_labels)), cap_means, yerr=cap_stds, 
                    capsize=5, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Capabilities')
    ax1.set_ylabel('Average Generalization Ratio')
    ax1.set_title('Generalization Ratio by Capability\n(Average across models)')
    ax1.set_xticks(range(len(cap_labels)))
    ax1.set_xticklabels(cap_labels, rotation=45, ha='right')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Generalization')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars1, cap_means)):
        ax1.annotate(f'{mean_val:.3f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", 
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Model comparison
    models = list(model_averages.keys())
    model_overall_means = []
    
    for model in models:
        model_values = list(model_averages[model].values())
        model_overall_means.append(np.mean(model_values) if model_values else 0)
    
    bars2 = ax2.bar(range(len(models)), model_overall_means, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Overall Average Generalization Ratio')
    ax2.set_title('Overall Generalization by Model\n(Average across all capabilities)')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace('-', '-\n').replace('_', '_\n') for m in models], ha='center')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Generalization')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars2, model_overall_means)):
        ax2.annotate(f'{mean_val:.3f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = f"{output_dir}/aggregate_generalization_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Aggregate comparison saved to: {output_path}")
    
    return model_averages

def create_capability_breakdown(all_data: Dict, output_dir: str):
    """Create a detailed breakdown showing all contexts within each capability."""
    
    # Collect all data by capability
    capability_data = {}
    
    for model_name, model_data in all_data.items():
        for capability, capability_info in model_data.items():
            if capability not in capability_data:
                capability_data[capability] = {}
            
            if 'contexts' in capability_info:
                for context, context_data in capability_info['contexts'].items():
                    if context not in capability_data[capability]:
                        capability_data[capability][context] = {}
                    
                    if 'generalization_ratio' in context_data:
                        capability_data[capability][context][model_name] = context_data['generalization_ratio']
    
    # Create one plot per capability
    capabilities = sorted(capability_data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, capability in enumerate(capabilities):
        if i >= len(axes):
            break
        
        ax = axes[i]
        contexts = sorted(capability_data[capability].keys())
        
        # Calculate averages across models for each context
        context_means = []
        context_labels = []
        
        for context in contexts:
            values = list(capability_data[capability][context].values())
            if values:
                context_means.append(np.mean(values))
                context_labels.append(context.replace('_', ' ').title()[:20] + '...' if len(context) > 20 else context.replace('_', ' ').title())
        
        bars = ax.bar(range(len(context_labels)), context_means, alpha=0.7)
        ax.set_title(f'{capability.title()}')
        ax.set_xlabel('Contexts')
        ax.set_ylabel('Avg Generalization Ratio')
        ax.set_xticks(range(len(context_labels)))
        ax.set_xticklabels(context_labels, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, context_means):
            ax.annotate(f'{mean_val:.2f}', 
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 2), textcoords="offset points", 
                       ha='center', va='bottom', fontsize=7)
    
    # Hide unused subplots
    for i in range(len(capabilities), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_path = f"{output_dir}/capability_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Capability breakdown saved to: {output_path}")

def create_per_model_comparison(all_data: Dict, output_dir: str):
    """Create separate comparison plots for each model across capabilities."""
    
    # Calculate averages for each model
    model_averages = {}
    for model_name, model_data in all_data.items():
        model_averages[model_name] = calculate_capability_averages(model_data)
    
    # Get all capabilities
    capabilities = set()
    for model_data in model_averages.values():
        capabilities.update(model_data.keys())
    capabilities = sorted(list(capabilities))
    
    # Create separate plots for each model
    models = list(model_averages.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = model_averages[model]
        
        # Get values for this model
        cap_values = []
        cap_labels = []
        
        for cap in capabilities:
            if cap in model_data:
                cap_values.append(model_data[cap])
                cap_labels.append(cap.title())
        
        bars = ax.bar(range(len(cap_labels)), cap_values, alpha=0.7, color='lightsteelblue')
        ax.set_xlabel('Capabilities')
        ax.set_ylabel('Average Generalization Ratio')
        ax.set_title(f'{model.replace("-", "-\\n")}')
        ax.set_xticks(range(len(cap_labels)))
        ax.set_xticklabels(cap_labels, rotation=45, ha='right')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Generalization')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set consistent y-axis limits
        ax.set_ylim(0.6, 1.0)
        
        # Add value labels
        for bar, val in zip(bars, cap_values):
            ax.annotate(f'{val:.3f}', 
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", 
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = f"{output_dir}/per_model_capability_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-model comparison saved to: {output_path}")

def main():
    """Main function to create aggregate visualizations."""
    base_dir = "/Users/ishaagarwal/Documents/ERA/multi-view-capabilities/experiment_pipeline/results"
    
    # Load all generalization data
    all_data = {}
    try:
        with open(f"{base_dir}/all_generalization_ratios.json", 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print("Error: all_generalization_ratios.json not found. Run generalization_ratio_analysis.py first.")
        return
    
    print("Creating aggregate visualizations...")
    
    # Create aggregate comparison
    model_averages = create_aggregate_comparison(all_data, base_dir)
    
    # Create per-model comparison
    create_per_model_comparison(all_data, base_dir)
    
    # Create capability breakdown
    create_capability_breakdown(all_data, base_dir)
    
    # Save summary statistics
    summary_stats = {
        "model_averages": model_averages,
        "capability_rankings": {}
    }
    
    # Rank capabilities by average generalization
    all_capabilities = set()
    for model_data in model_averages.values():
        all_capabilities.update(model_data.keys())
    
    for cap in all_capabilities:
        values = []
        for model_data in model_averages.values():
            if cap in model_data:
                values.append(model_data[cap])
        
        if values:
            summary_stats["capability_rankings"][cap] = {
                "average": np.mean(values),
                "std": np.std(values) if len(values) > 1 else 0,
                "models_count": len(values)
            }
    
    # Sort by average
    sorted_capabilities = sorted(summary_stats["capability_rankings"].items(), 
                               key=lambda x: x[1]["average"], reverse=True)
    
    print("\n--- SUMMARY STATISTICS ---")
    print("Capabilities ranked by average generalization ratio:")
    for i, (cap, stats) in enumerate(sorted_capabilities, 1):
        print(f"{i}. {cap.title()}: {stats['average']:.3f} (±{stats['std']:.3f})")
    
    # Save summary
    with open(f"{base_dir}/generalization_summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nSummary statistics saved to: {base_dir}/generalization_summary_stats.json")

if __name__ == "__main__":
    main()