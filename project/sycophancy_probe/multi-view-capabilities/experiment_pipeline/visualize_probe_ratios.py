#!/usr/bin/env python3
"""
Visualize probe ratios to highlight shared trends between capabilities.
Creates multiple intuitive visualizations to reveal patterns across capabilities.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_all_model_data(base_path="results"):
    """Load all model ratio data."""
    model_data = {}
    
    # Look for probe ratio files in model directories
    for ratio_file in glob.glob(f"{base_path}/*/*_probe_ratios.json"):
        model_name = Path(ratio_file).stem.replace("_probe_ratios", "")
        
        with open(ratio_file, 'r') as f:
            model_data[model_name] = json.load(f)
    
    return model_data

def create_heatmap_with_clustering(model_data, model_name, output_dir):
    """Create a heatmap showing ratios across capabilities and contexts."""
    data = model_data[model_name]
    
    # Create DataFrame
    all_contexts = set()
    for cap_data in data.values():
        all_contexts.update(cap_data.keys())
    
    all_contexts = sorted(list(all_contexts))
    capabilities = sorted(data.keys())
    
    # Build matrix
    matrix = np.zeros((len(capabilities), len(all_contexts)))
    for i, cap in enumerate(capabilities):
        for j, context in enumerate(all_contexts):
            if context in data[cap]:
                matrix[i, j] = data[cap][context]
            else:
                matrix[i, j] = np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Mask NaN values
    mask = np.isnan(matrix)
    
    sns.heatmap(matrix, 
                xticklabels=[ctx.replace('_', ' ').title() for ctx in all_contexts],
                yticklabels=[cap.replace('_', ' ').title() for cap in capabilities],
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.95,
                mask=mask,
                cbar_kws={'label': 'General/Context Accuracy Ratio'},
                ax=ax)
    
    plt.title(f'{model_name}: Probe Accuracy Ratios Across Capabilities', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Contexts', fontsize=12, fontweight='bold')
    plt.ylabel('Capabilities', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_comparison(model_data, model_name, output_dir):
    """Create violin plots showing ratio distributions by capability."""
    data = model_data[model_name]
    
    # Prepare data for plotting
    plot_data = []
    for capability, contexts in data.items():
        for context, ratio in contexts.items():
            plot_data.append({
                'Capability': capability.replace('_', ' ').title(),
                'Ratio': ratio,
                'Context': context.replace('_', ' ').title()
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create violin plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Violin plot
    sns.violinplot(data=df, x='Capability', y='Ratio', ax=ax1, inner='box')
    ax1.set_title(f'{model_name}: Ratio Distributions by Capability', 
                  fontsize=14, fontweight='bold')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Ratio (1.0)')
    ax1.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% Ratio')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Box plot for clearer quartiles
    sns.boxplot(data=df, x='Capability', y='Ratio', ax=ax2)
    ax2.set_title('Quartile View', fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_capability_similarity_matrix(model_data, model_name, output_dir):
    """Create correlation matrix between capabilities."""
    data = model_data[model_name]
    
    # Get all unique contexts
    all_contexts = set()
    for cap_data in data.values():
        all_contexts.update(cap_data.keys())
    all_contexts = sorted(list(all_contexts))
    
    capabilities = sorted(data.keys())
    
    # Create correlation matrix
    correlations = np.zeros((len(capabilities), len(capabilities)))
    
    for i, cap1 in enumerate(capabilities):
        for j, cap2 in enumerate(capabilities):
            if i == j:
                correlations[i, j] = 1.0
            else:
                # Get common contexts
                common_contexts = set(data[cap1].keys()) & set(data[cap2].keys())
                if len(common_contexts) > 1:
                    values1 = [data[cap1][ctx] for ctx in common_contexts]
                    values2 = [data[cap2][ctx] for ctx in common_contexts]
                    corr, _ = pearsonr(values1, values2)
                    correlations[i, j] = corr if not np.isnan(corr) else 0
                else:
                    correlations[i, j] = 0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlations,
                xticklabels=[cap.replace('_', ' ').title() for cap in capabilities],
                yticklabels=[cap.replace('_', ' ').title() for cap in capabilities],
                annot=True,
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax)
    
    plt.title(f'{model_name}: Capability Similarity Matrix\n(Based on Ratio Patterns)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_similarity_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_ratio_range_analysis(model_data, model_name, output_dir):
    """Create stacked bar chart showing ratio range distributions."""
    data = model_data[model_name]
    
    # Define ratio ranges
    ranges = [
        (0.0, 0.85, '< 0.85'),
        (0.85, 0.90, '0.85-0.90'),
        (0.90, 0.95, '0.90-0.95'),
        (0.95, 1.0, '0.95-1.00'),
        (1.0, float('inf'), '> 1.00')
    ]
    
    # Calculate proportions for each capability
    capabilities = sorted(data.keys())
    range_counts = {range_name: [] for _, _, range_name in ranges}
    
    for capability in capabilities:
        ratios = list(data[capability].values())
        total_contexts = len(ratios)
        
        for min_val, max_val, range_name in ranges:
            if max_val == float('inf'):
                count = sum(1 for r in ratios if r >= min_val)
            else:
                count = sum(1 for r in ratios if min_val <= r < max_val)
            
            range_counts[range_name].append(count / total_contexts)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bottom = np.zeros(len(capabilities))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(ranges)))
    
    for i, (_, _, range_name) in enumerate(ranges):
        values = range_counts[range_name]
        bars = ax.bar(capabilities, values, bottom=bottom, 
                     label=range_name, color=colors[i], alpha=0.8)
        bottom += values
        
        # Add percentage labels for significant segments
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.1:  # Only show labels for segments > 10%
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       bottom[j] - height/2,
                       f'{val:.1%}', ha='center', va='center',
                       fontweight='bold', color='white' if val > 0.3 else 'black')
    
    ax.set_title(f'{model_name}: Ratio Range Distribution by Capability', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Capabilities', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion of Contexts', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Format capability names
    ax.set_xticklabels([cap.replace('_', ' ').title() for cap in capabilities], 
                       rotation=45, ha='right')
    
    ax.legend(title='Ratio Range', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_ratio_ranges.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_pca_visualization(model_data, output_dir):
    """Create PCA visualization of capabilities across models."""
    # Combine data from all models
    all_capabilities = []
    all_vectors = []
    all_models = []
    
    # Get all unique contexts across all models
    all_contexts = set()
    for model_data_dict in model_data.values():
        for cap_data in model_data_dict.values():
            all_contexts.update(cap_data.keys())
    all_contexts = sorted(list(all_contexts))
    
    for model_name, data in model_data.items():
        for capability, contexts in data.items():
            # Create vector for this capability
            vector = []
            for context in all_contexts:
                if context in contexts:
                    vector.append(contexts[context])
                else:
                    vector.append(np.nan)
            
            all_capabilities.append(f"{model_name}_{capability}")
            all_vectors.append(vector)
            all_models.append(model_name)
    
    # Create DataFrame and handle missing values
    df = pd.DataFrame(all_vectors, columns=all_contexts, index=all_capabilities)
    df = df.fillna(df.mean())  # Fill NaN with column means
    
    # Perform PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by model
    unique_models = list(set(all_models))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_models)))
    model_colors = {model: colors[i] for i, model in enumerate(unique_models)}
    
    # Keep track of which models we've already added to legend
    legend_models = set()
    
    for i, (cap, model) in enumerate(zip(all_capabilities, all_models)):
        capability_name = cap.split('_', 1)[1].replace('_', ' ').title()
        
        # Only add label for first occurrence of each model
        label = model if model not in legend_models else ""
        if model not in legend_models:
            legend_models.add(model)
            
        ax.scatter(pca_result[i, 0], pca_result[i, 1], 
                  c=[model_colors[model]], s=100, alpha=0.7,
                  label=label)
        ax.annotate(capability_name, (pca_result[i, 0], pca_result[i, 1]),
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=9, alpha=0.8)
    
    ax.set_title('PCA: Capability Clustering Based on Ratio Patterns', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=12)
    
    # Create custom legend
    handles = [plt.scatter([], [], c=color, s=100, alpha=0.7, label=model) 
               for model, color in model_colors.items()]
    ax.legend(handles=handles, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_capability_clustering.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics(model_data, output_dir):
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
    
    print("Summary Statistics:")
    print(df.round(4).to_string(index=False))

def create_simple_bar_charts(model_data, output_dir):
    """Create simple, clean bar charts showing key insights."""
    
    # Prepare data for plotting
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
    
    # 1. Simple bar chart of average ratios by capability (all models)
    plt.figure(figsize=(12, 6))
    
    # Group by capability and calculate overall mean
    cap_means = df.groupby('Capability')['Mean_Ratio'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(cap_means)), cap_means.values, 
                   color='steelblue', alpha=0.7, edgecolor='navy')
    
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, 
                label='Perfect Ratio (1.0)')
    plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.8, 
                label='95% Threshold')
    
    plt.title('Average Probe Accuracy Ratios by Capability\n(General vs Context-Specific)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Capabilities', fontsize=12, fontweight='bold')
    plt.ylabel('Average Ratio', fontsize=12, fontweight='bold')
    
    plt.xticks(range(len(cap_means)), cap_means.index, rotation=45, ha='right')
    plt.ylim(0.8, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/simple_capability_averages.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Model comparison chart
    if len(model_data) > 1:
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        capabilities_unique = sorted(df['Capability'].unique())
        models_unique = sorted(df['Model'].unique())
        
        x = np.arange(len(capabilities_unique))
        width = 0.35
        
        for i, model in enumerate(models_unique):
            model_data_subset = df[df['Model'] == model]
            ratios = [model_data_subset[model_data_subset['Capability'] == cap]['Mean_Ratio'].iloc[0] 
                     if cap in model_data_subset['Capability'].values else 0 
                     for cap in capabilities_unique]
            
            bars = plt.bar(x + i * width, ratios, width, label=model, alpha=0.8)
            
            # Add value labels
            for bar, ratio in zip(bars, ratios):
                if ratio > 0:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, label='Perfect Ratio')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.8, label='95% Threshold')
        
        plt.title('Model Comparison: Average Ratios by Capability', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Capabilities', fontsize=12, fontweight='bold')
        plt.ylabel('Average Ratio', fontsize=12, fontweight='bold')
        
        plt.xticks(x + width/2, capabilities_unique, rotation=45, ha='right')
        plt.ylim(0.8, 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison_simple.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Summary insights chart
    plt.figure(figsize=(10, 6))
    
    # Calculate key metrics for each capability
    insights_data = []
    for capability in df['Capability'].unique():
        cap_data = df[df['Capability'] == capability]
        insights_data.append({
            'Capability': capability,
            'Mean': cap_data['Mean_Ratio'].mean(),
            'Above_1.0': (cap_data['Mean_Ratio'] > 1.0).sum(),
            'Below_0.95': (cap_data['Mean_Ratio'] < 0.95).sum()
        })
    
    insights_df = pd.DataFrame(insights_data).sort_values('Mean', ascending=True)
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(insights_df)), insights_df['Mean'], 
                    color='lightcoral', alpha=0.7, edgecolor='darkred')
    
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='Perfect Ratio')
    plt.axvline(x=0.95, color='orange', linestyle='--', alpha=0.8, label='95% Threshold')
    
    plt.title('Capability Performance Summary\n(Higher = General performs better relative to specialized)', 
              fontsize=12, fontweight='bold', pad=20)
    plt.xlabel('Average Ratio', fontsize=11, fontweight='bold')
    plt.ylabel('Capabilities', fontsize=11, fontweight='bold')
    
    plt.yticks(range(len(insights_df)), insights_df['Capability'])
    plt.xlim(0.85, 1.05)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/capability_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory - using relative path from experiment_pipeline
    output_dir = "ratio_visualizations"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load all model data
    model_data = load_all_model_data()
    
    if not model_data:
        print("No model data found!")
        return
    
    print(f"Found data for models: {list(model_data.keys())}")
    
    # Create simple, intuitive visualizations
    print("\nCreating simple visualizations...")
    create_simple_bar_charts(model_data, output_dir)
    print("  ✓ Simple bar charts created")
    
    # Create summary statistics
    print("\nGenerating summary statistics...")
    create_summary_statistics(model_data, output_dir)
    print("  ✓ Summary statistics saved")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nKey files created:")
    print("  - simple_capability_averages.png: Average ratios by capability")
    print("  - model_comparison_simple.png: Side-by-side model comparison") 
    print("  - capability_performance_summary.png: Horizontal bar summary")
    print("  - summary_statistics.csv: Detailed statistics table")

if __name__ == "__main__":
    main()