import os
import json
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from typing import Dict, Any, Tuple, List
from scipy.stats import ttest_ind

# Import LinearProbe class so we can load saved probe objects
# The saved probes reference this class, so we need to make it available
try:
    from linear_probe import LinearProbe
except ImportError:
    # Define it here if import fails, matching the structure from linear_probe.py
    class LinearProbe(nn.Module):
        def __init__(self, input_features, output_features=1):
            super(LinearProbe, self).__init__()
            self.linear = nn.Linear(input_features, output_features)

        def forward(self, x):
            return self.linear(x).squeeze(-1)

def load_steering_vectors(svs_dir="experiment1/svs"):
    """
    Load all steering vectors from the directory structure.
    Returns: {capability: {context_folder: [(vector_tensor, filename), ...]}}
    """
    svs_dir = Path(svs_dir)
    vectors_by_capability = {}
    
    if not svs_dir.exists():
        raise ValueError(f"Directory {svs_dir} does not exist")
    
    for capability_dir in svs_dir.iterdir():
        if not capability_dir.is_dir():
            continue
        
        capability_name = capability_dir.name
        vectors_by_capability[capability_name] = {}
        
        for context_dir in capability_dir.iterdir():
            if not context_dir.is_dir():
                continue
            
            context_name = context_dir.name
            vectors_by_capability[capability_name][context_name] = []
            
            for sv_file in context_dir.glob("*.pt"):
                try:
                    vector = torch.load(sv_file, map_location="cpu", weights_only=False)
                    if isinstance(vector, torch.Tensor):
                        # Store full path for uniqueness, but also keep filename for display
                        vectors_by_capability[capability_name][context_name].append(
                            (vector, sv_file.name, str(sv_file))
                        )
                except Exception as e:
                    print(f"Warning: Could not load {sv_file}: {e}")
                    continue
    
    return vectors_by_capability

def compute_cosine_similarity_matrix(vectors):
    """
    Compute pairwise cosine similarity matrix for a list of vectors.
    
    Args:
        vectors: List of (vector_tensor, filename) tuples
    
    Returns:
        similarity_matrix: numpy array of shape (n, n)
        labels: List of labels for each vector
    """
    if len(vectors) == 0:
        return np.array([]), []
    
    if len(vectors) == 1:
        return np.array([[1.0]]), [vectors[0][1]]
    
    # Extract tensors and labels, detach and convert to float32 to avoid BFloat16 and grad issues
    vector_tensors = [v[0].detach().to(torch.float32) for v in vectors]
    labels = [v[1] for v in vectors]
    
    # Stack vectors into a tensor
    stacked_vectors = torch.stack(vector_tensors)
    
    # Normalize vectors
    normalized = torch.nn.functional.normalize(stacked_vectors, p=2, dim=1)
    
    # Compute cosine similarity matrix (dot product of normalized vectors)
    similarity_matrix = torch.mm(normalized, normalized.t())
    
    # Convert to numpy
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    
    return similarity_matrix_np, labels

def load_general_vectors(svs_dir):
    """
    Load all general vectors (files starting with 'gen') from all capability folders.
    Returns: List of (vector_tensor, label) tuples where label includes capability and context info
    """
    svs_dir = Path(svs_dir)
    general_vectors = []
    
    if not svs_dir.exists():
        raise ValueError(f"Directory {svs_dir} does not exist")
    
    for capability_dir in svs_dir.iterdir():
        if not capability_dir.is_dir():
            continue
        
        capability_name = capability_dir.name
        
        for context_dir in capability_dir.iterdir():
            if not context_dir.is_dir():
                continue
            
            context_name = context_dir.name
            
            for sv_file in context_dir.glob("*.pt"):
                # Check if filename starts with 'gen' (case-insensitive)
                if sv_file.name.lower().startswith('gen'):
                    try:
                        vector = torch.load(sv_file, map_location="cpu", weights_only=False)
                        if isinstance(vector, torch.Tensor):
                            # Create label with capability and context info
                            label = f"{capability_name}/{context_name}/{sv_file.name}"
                            general_vectors.append((vector, label))
                    except Exception as e:
                        print(f"Warning: Could not load {sv_file}: {e}")
                        continue
    
    return general_vectors

def compute_gen_cosine_similarity_cross_capability(svs_dir, output_dir):
    """
    Find all general vectors across all capabilities, compute cross-capability 
    cosine similarity matrix, and generate a heatmap.
    
    Args:
        svs_dir: Directory containing steering vectors
        output_dir: Directory to save the heatmap
    """
    print("\nLoading general vectors from all capabilities...")
    general_vectors = load_general_vectors(svs_dir)
    
    if len(general_vectors) == 0:
        print("No general vectors found!")
        return
    
    if len(general_vectors) == 1:
        print(f"Only 1 general vector found: {general_vectors[0][1]}")
        print("Need at least 2 vectors for similarity comparison.")
        return
    
    print(f"Found {len(general_vectors)} general vectors:")
    for _, label in general_vectors:
        print(f"  {label}")
    
    print("\nComputing cross-capability cosine similarity matrix...")
    similarity_matrix, labels = compute_cosine_similarity_matrix(general_vectors)
    
    # Create a more readable heatmap with capability names
    print("\nGenerating cross-capability similarity heatmap...")
    create_cross_capability_heatmap(similarity_matrix, labels, output_dir)
    
    # Calculate and print average similarity
    avg_similarity = calculate_average_similarity(similarity_matrix)
    print(f"\nAverage cosine similarity across all general vectors: {avg_similarity:.4f}")
    
    return similarity_matrix, labels

def create_cross_capability_heatmap(similarity_matrix, labels, output_dir):
    """
    Create a heatmap for cross-capability general vector similarities.
    
    Args:
        similarity_matrix: numpy array of shape (n, n)
        labels: List of full path labels (capability/context/filename)
        output_dir: Directory to save the heatmap
    """
    if similarity_matrix.size == 0:
        print("Skipping cross-capability heatmap: no vectors found")
        return
    
    plt.figure(figsize=(max(12, len(labels) * 0.9), max(10, len(labels) * 0.8)))
    
    # Extract capability names and create readable labels
    readable_labels = []
    for label in labels:
        parts = label.split('/')
        if len(parts) >= 2:
            capability = parts[0]
            filename = parts[-1]
            # Shorten filename for readability
            short_filename = filename.replace("_bipo_l15_", "_l15_").replace("general", "gen")[:30]
            readable_labels.append(f"{capability}\n{short_filename}")
        else:
            readable_labels.append(label[:40])
    
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=readable_labels,
        yticklabels=readable_labels,
        cbar_kws={"label": "Cosine Similarity"},
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5
    )
    
    plt.title("Cross-Capability Cosine Similarity: General Vectors", fontsize=16, fontweight="bold")
    plt.xlabel("General Steering Vector", fontsize=12)
    plt.ylabel("General Steering Vector", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / "cross_capability_general_vectors_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved cross-capability heatmap to {output_path}")

def calculate_average_similarity(similarity_matrix):
    """
    Calculate average pairwise similarity from upper triangle (excluding diagonal).
    
    Args:
        similarity_matrix: numpy array of shape (n, n)
    
    Returns:
        average_similarity: float
    """
    if similarity_matrix.size == 0:
        return 0.0
    
    if similarity_matrix.shape[0] == 1:
        return 1.0
    
    # Get upper triangle indices (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    upper_triangle_values = similarity_matrix[upper_triangle_indices]
    
    return np.mean(upper_triangle_values)

def create_heatmap(similarity_matrix, labels, capability_name, output_dir):
    """
    Create and save a heatmap visualization of the similarity matrix.
    
    Args:
        similarity_matrix: numpy array of shape (n, n)
        labels: List of labels for each vector
        capability_name: Name of the capability
        output_dir: Directory to save the heatmap
    """
    if similarity_matrix.size == 0:
        print(f"Skipping heatmap for {capability_name}: no vectors found")
        return
    
    plt.figure(figsize=(max(10, len(labels) * 0.8), max(8, len(labels) * 0.7)))
    
    # Create shortened labels for readability while preserving uniqueness
    # Extract timestamp or unique identifier from filename to keep labels distinct
    short_labels = []
    for label in labels:
        # Try to extract timestamp (format: YYYYMMDD_HHMMSS) from the label
        parts = label.split('/')
        if len(parts) >= 2:
            filename = parts[-1]
            # Extract timestamp if present (look for pattern like 20250822_223407)
            timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                # Create short label with context and timestamp
                context = parts[0]
                short_label = f"{context[:20]}/...{timestamp[-6:]}"  # Last 6 digits of timestamp
            else:
                # Fallback: use first 50 chars but ensure we include unique parts
                short_label = label.replace("_bipo_l15_", "_l15_").replace("specific", "spec_").replace("general", "gen_")[:50]
        else:
            short_label = label[:50]
        short_labels.append(short_label)
    
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar_kws={"label": "Cosine Similarity"},
        vmin=0.0,
        vmax=1.0
    )
    
    plt.title(f"Cosine Similarity Matrix: {capability_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Steering Vector", fontsize=12)
    plt.ylabel("Steering Vector", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / f"{capability_name}_similarity_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved heatmap to {output_path}")

def create_bar_chart(capability_averages, output_dir):
    """
    Create and save a bar chart showing average similarity per capability.
    
    Args:
        capability_averages: Dict of {capability: average_similarity}
        output_dir: Directory to save the chart
    """
    if not capability_averages:
        print("No capability averages to plot")
        return
    
    capabilities = list(capability_averages.keys())
    averages = [capability_averages[cap] for cap in capabilities]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(capabilities, averages, color="steelblue", edgecolor="navy", alpha=0.7)
    
    # Add value labels on bars
    for bar, avg in zip(bars, averages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{avg:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    
    plt.title("Average Cosine Similarity Within Each Capability", fontsize=14, fontweight="bold")
    plt.xlabel("Capability", fontsize=12)
    plt.ylabel("Average Cosine Similarity", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(averages) * 1.15 if averages else 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "capability_averages.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved bar chart to {output_path}")

def save_results(results, output_dir):
    """
    Save similarity results to JSON file.
    
    Args:
        results: Dict with structure {capability: {matrix: [...], average: float, vector_labels: [...]}}
        output_dir: Directory to save the JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for capability, data in results.items():
        json_results[capability] = {
            "matrix": data["matrix"].tolist() if isinstance(data["matrix"], np.ndarray) else data["matrix"],
            "average": float(data["average"]),
            "vector_labels": data["vector_labels"]
        }
    
    output_path = output_dir / "similarity_results.json"
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Saved results to {output_path}")


# Probe weight analysis functions
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

def load_probe_results_structure(results_dir):
    """
    Load the structure of probe results, finding all capability and context folders.
    
    Args:
        results_dir: Path to results directory (e.g., experiment_pipeline/results/Qwen2.5-7B-Instruct)
    
    Returns:
        Dict mapping capability -> list of context names
    """
    results_dir = Path(results_dir)
    structure = {}
    
    if not results_dir.exists():
        raise ValueError(f"Directory {results_dir} does not exist")
    
    for capability_dir in results_dir.iterdir():
        if not capability_dir.is_dir():
            continue
        
        capability_name = capability_dir.name
        structure[capability_name] = []
        
        for context_dir in capability_dir.iterdir():
            if not context_dir.is_dir():
                continue
            
            # Skip cross_layer_analysis folders
            if context_dir.name == "cross_layer_analysis":
                continue
            
            context_name = context_dir.name
            structure[capability_name].append(context_name)
    
    return structure

def find_best_layers_for_contexts(results_dir, save_path=None):
    """
    Find the best performing layer for each context in each capability.
    Uses find_best_layer() from analyze_probe_ratios.py logic.
    
    Args:
        results_dir: Path to results directory
        save_path: Optional path to save the best layers mapping as JSON
    
    Returns:
        Dict mapping capability -> context -> best_layer_name
    """
    results_dir = Path(results_dir)
    best_layers_map = {}
    
    structure = load_probe_results_structure(results_dir)
    
    print("Finding best layers for each context...")
    for capability_name, contexts in structure.items():
        best_layers_map[capability_name] = {}
        
        for context_name in contexts:
            context_dir = results_dir / capability_name / context_name
            summary_file = context_dir / "all_layers_summary.json"
            
            if not summary_file.exists():
                print(f"Warning: {summary_file} not found, skipping {capability_name}/{context_name}")
                continue
            
            try:
                with open(summary_file, 'r') as f:
                    all_layers_data = json.load(f)
                
                # Determine target context string - format matches what's in the JSON
                if context_name == "general":
                    target_context = "general (in-context)"
                else:
                    target_context = f"{context_name} (in-context)"
                
                best_layer, best_accuracy = find_best_layer(all_layers_data, target_context)
                
                if best_layer:
                    best_layers_map[capability_name][context_name] = best_layer
                    print(f"  {capability_name}/{context_name}: best layer = {best_layer} (accuracy = {best_accuracy:.4f})")
                else:
                    print(f"  Warning: No best layer found for {capability_name}/{context_name} with target '{target_context}'")
                    
            except Exception as e:
                print(f"Error processing {summary_file}: {e}")
                continue
    
    # Save the mapping to a JSON file
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(best_layers_map, f, indent=2)
        print(f"\nSaved best layers mapping to {save_path}")
    
    return best_layers_map

def load_best_layer_probe_weights(results_dir, best_layers_map):
    """
    Load probe weights from the best layers for each context.
    
    Args:
        results_dir: Path to results directory
        best_layers_map: Dict mapping capability -> context -> best_layer_name
    
    Returns:
        Dict mapping capability -> list of (probe_weight_tensor, label) tuples
    """
    results_dir = Path(results_dir)
    probe_weights_by_capability = {}
    
    for capability_name, contexts_map in best_layers_map.items():
        probe_weights_by_capability[capability_name] = []
        
        for context_name, best_layer in contexts_map.items():
            layer_dir = results_dir / capability_name / context_name / best_layer
            probe_weights_file = layer_dir / "training_data" / "probe_weights.pt"
            
            if not probe_weights_file.exists():
                print(f"Warning: {probe_weights_file} not found, skipping")
                continue
            
            try:
                # Load with custom unpickler to handle LinearProbe class reference
                # Make LinearProbe available in the current module namespace for unpickling
                import sys
                import importlib.util
                
                # Try to load the probe, making LinearProbe available in __main__ if needed
                if '__main__' not in sys.modules or not hasattr(sys.modules['__main__'], 'LinearProbe'):
                    # Add LinearProbe to __main__ module for unpickling
                    import __main__
                    __main__.LinearProbe = LinearProbe
                
                probe = torch.load(probe_weights_file, map_location="cpu", weights_only=False)
                
                # Extract weight tensor from LinearProbe object
                if hasattr(probe, 'linear') and hasattr(probe.linear, 'weight'):
                    # It's a LinearProbe object
                    weight_tensor = probe.linear.weight.squeeze(0)  # Remove output dimension if it's 1
                elif isinstance(probe, torch.Tensor):
                    # It's already a tensor
                    weight_tensor = probe.squeeze(0) if probe.dim() > 1 else probe
                else:
                    print(f"Warning: Unknown probe structure in {probe_weights_file}")
                    continue
                
                # Detach from computation graph and convert to float32 to avoid BFloat16 issues
                weight_tensor = weight_tensor.detach().to(torch.float32)
                
                # Create label
                label = f"{capability_name}/{context_name}/{best_layer}"
                probe_weights_by_capability[capability_name].append((weight_tensor, label))
                
            except Exception as e:
                print(f"Error loading {probe_weights_file}: {e}")
                continue
    
    return probe_weights_by_capability

def create_probe_heatmap(similarity_matrix, labels, capability_name, output_dir):
    """
    Create and save a heatmap visualization for probe weights similarity matrix.
    Similar to create_heatmap but with probe-specific labels.
    
    Args:
        similarity_matrix: numpy array of shape (n, n)
        labels: List of labels for each probe weight (format: capability/context/layer)
        capability_name: Name of the capability
        output_dir: Directory to save the heatmap
    """
    if similarity_matrix.size == 0:
        print(f"Skipping heatmap for {capability_name}: no probe weights found")
        return
    
    plt.figure(figsize=(max(10, len(labels) * 0.8), max(8, len(labels) * 0.7)))
    
    # Create shortened labels for readability
    short_labels = []
    for label in labels:
        parts = label.split('/')
        if len(parts) >= 3:
            context = parts[1]
            layer = parts[2]
            # Shorten context name and keep layer info
            short_label = f"{context[:20]}/{layer}"
        else:
            short_label = label[:50]
        short_labels.append(short_label)
    
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar_kws={"label": "Cosine Similarity"},
        vmin=0.0,
        vmax=1.0
    )
    
    plt.title(f"Probe Weight Cosine Similarity Matrix: {capability_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Probe Weight", fontsize=12)
    plt.ylabel("Probe Weight", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / f"{capability_name}_probe_similarity_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved probe heatmap to {output_path}")

def load_general_probe_weights(results_dir, best_layers_map):
    """
    Load all general probe weights (from "general" context) from all capability folders.
    
    Args:
        results_dir: Path to results directory
        best_layers_map: Dict mapping capability -> context -> best_layer_name
    
    Returns:
        List of (probe_weight_tensor, label) tuples where label includes capability info
    """
    results_dir = Path(results_dir)
    general_probe_weights = []
    
    for capability_name, contexts_map in best_layers_map.items():
        # Look for "general" context in this capability
        if "general" in contexts_map:
            best_layer = contexts_map["general"]
            layer_dir = results_dir / capability_name / "general" / best_layer
            probe_weights_file = layer_dir / "training_data" / "probe_weights.pt"
            
            if not probe_weights_file.exists():
                print(f"Warning: {probe_weights_file} not found, skipping")
                continue
            
            try:
                # Load with LinearProbe class available
                import sys
                if '__main__' not in sys.modules or not hasattr(sys.modules['__main__'], 'LinearProbe'):
                    import __main__
                    __main__.LinearProbe = LinearProbe
                
                probe = torch.load(probe_weights_file, map_location="cpu", weights_only=False)
                
                # Extract weight tensor from LinearProbe object
                if hasattr(probe, 'linear') and hasattr(probe.linear, 'weight'):
                    weight_tensor = probe.linear.weight.squeeze(0)
                elif isinstance(probe, torch.Tensor):
                    weight_tensor = probe.squeeze(0) if probe.dim() > 1 else probe
                else:
                    print(f"Warning: Unknown probe structure in {probe_weights_file}")
                    continue
                
                # Detach and convert to float32
                weight_tensor = weight_tensor.detach().to(torch.float32)
                
                # Create label with capability and layer info
                label = f"{capability_name}/general/{best_layer}"
                general_probe_weights.append((weight_tensor, label))
                
            except Exception as e:
                print(f"Error loading {probe_weights_file}: {e}")
                continue
    
    return general_probe_weights

def compute_gen_probe_cosine_similarity_cross_capability(results_dir, best_layers_map, output_dir):
    """
    Find all general probe weights across all capabilities, compute cross-capability 
    cosine similarity matrix, and generate a heatmap.
    
    Args:
        results_dir: Path to results directory
        best_layers_map: Dict mapping capability -> context -> best_layer_name
        output_dir: Directory to save the heatmap
    """
    print("\nLoading general probe weights from all capabilities...")
    general_probe_weights = load_general_probe_weights(results_dir, best_layers_map)
    
    if len(general_probe_weights) == 0:
        print("No general probe weights found!")
        return
    
    if len(general_probe_weights) == 1:
        print(f"Only 1 general probe weight found: {general_probe_weights[0][1]}")
        print("Need at least 2 probe weights for similarity comparison.")
        return
    
    print(f"Found {len(general_probe_weights)} general probe weights:")
    for _, label in general_probe_weights:
        print(f"  {label}")
    
    print("\nComputing cross-capability cosine similarity matrix...")
    similarity_matrix, labels = compute_cosine_similarity_matrix(general_probe_weights)
    
    # Create a more readable heatmap with capability names
    print("\nGenerating cross-capability similarity heatmap...")
    create_cross_capability_probe_heatmap(similarity_matrix, labels, output_dir)
    
    # Calculate and print average similarity
    avg_similarity = calculate_average_similarity(similarity_matrix)
    print(f"\nAverage cosine similarity across all general probe weights: {avg_similarity:.4f}")
    
    # Save cross-capability matrix for aggregation
    cross_capability_data = {
        "matrix": similarity_matrix.tolist() if isinstance(similarity_matrix, np.ndarray) else similarity_matrix,
        "labels": labels,
        "average": float(avg_similarity)
    }
    cross_capability_path = output_dir / "cross_capability_general_probe_weights.json"
    with open(cross_capability_path, 'w') as f:
        json.dump(cross_capability_data, f, indent=2)
    print(f"Saved cross-capability matrix to {cross_capability_path}")
    
    return similarity_matrix, labels

def create_cross_capability_probe_heatmap(similarity_matrix, labels, output_dir):
    """
    Create a heatmap for cross-capability general probe weight similarities.
    
    Args:
        similarity_matrix: numpy array of shape (n, n)
        labels: List of full path labels (capability/general/layer)
        output_dir: Directory to save the heatmap
    """
    if similarity_matrix.size == 0:
        print("Skipping cross-capability heatmap: no probe weights found")
        return
    
    plt.figure(figsize=(max(12, len(labels) * 0.9), max(10, len(labels) * 0.8)))
    
    # Extract capability names and create readable labels
    readable_labels = []
    for label in labels:
        parts = label.split('/')
        if len(parts) >= 3:
            capability = parts[0]
            layer = parts[2]
            readable_labels.append(f"{capability}\n{layer}")
        else:
            readable_labels.append(label[:40])
    
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=readable_labels,
        yticklabels=readable_labels,
        cbar_kws={"label": "Cosine Similarity"},
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5
    )
    
    plt.title("Cross-Capability Cosine Similarity: General Probe Weights", fontsize=16, fontweight="bold")
    plt.xlabel("General Probe Weight", fontsize=12)
    plt.ylabel("General Probe Weight", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / "cross_capability_general_probe_weights_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved cross-capability heatmap to {output_path}")

# Aggregation functions (same as steering vectors but adapted for probe weights)
def calculate_cross_capability_general_average(similarity_matrix):
    """
    Calculate average cross-capability cosine similarity from general probe weights matrix.
    Uses upper triangle excluding diagonal.
    
    Args:
        similarity_matrix: numpy array of shape (n, n)
    
    Returns:
        average_similarity: float
    """
    return calculate_average_similarity(similarity_matrix)

def find_general_probe_index(labels):
    """
    Find the index of the general probe weight in the labels list.
    For probe weights, format is: capability/general/layer
    
    Args:
        labels: List of probe weight labels
    
    Returns:
        index: int or None if not found
    """
    for i, label in enumerate(labels):
        # For probe weights: check if context part is "general"
        parts = label.split('/')
        if len(parts) >= 2 and parts[1].lower() == 'general':
            return i
    return None

def calculate_general_to_context_averages(results_dict):
    """
    Calculate average similarity between general probe weight and context-specific probe weights
    for each capability.
    
    Args:
        results_dict: Dict with structure {capability: {matrix: [...], vector_labels: [...]}}
    
    Returns:
        Dict mapping capability -> average_similarity
    """
    general_to_context_avgs = {}
    
    for capability, data in results_dict.items():
        matrix = np.array(data["matrix"])
        labels = data["vector_labels"]
        
        general_idx = find_general_probe_index(labels)
        
        if general_idx is None:
            print(f"Warning: No general probe weight found for {capability}, skipping")
            continue
        
        # Extract the row corresponding to general probe weight
        general_row = matrix[general_idx, :]
        
        # Filter out: diagonal entry (self-similarity) and any other general probe weights
        # Keep only similarities to context-specific probe weights
        context_similarities = []
        for i, similarity in enumerate(general_row):
            if i == general_idx:
                continue  # Skip self-similarity
            # Check if this is another general probe weight
            parts = labels[i].split('/')
            if len(parts) >= 2 and parts[1].lower() == 'general':
                continue  # Skip other general probe weights
            context_similarities.append(similarity)
        
        if context_similarities:
            avg_sim = np.mean(context_similarities)
            general_to_context_avgs[capability] = float(avg_sim)
        else:
            print(f"Warning: No context-specific probe weights found for {capability}")
    
    return general_to_context_avgs

def calculate_context_to_context_averages(results_dict):
    """
    Calculate average similarity between context-specific probe weights only,
    completely excluding the general probe weight (both row and column).
    
    Args:
        results_dict: Dict with structure {capability: {matrix: [...], vector_labels: [...]}}
    
    Returns:
        Dict mapping capability -> average_similarity
    """
    context_to_context_avgs = {}
    
    for capability, data in results_dict.items():
        matrix = np.array(data["matrix"])
        labels = data["vector_labels"]
        
        general_idx = find_general_probe_index(labels)
        
        if general_idx is None:
            print(f"Warning: No general probe weight found for {capability}, skipping")
            continue
        
        # Remove the row and column corresponding to general probe weight
        # Create a sub-matrix without general probe weight
        context_indices = [i for i in range(len(labels)) if i != general_idx]
        
        if len(context_indices) < 2:
            print(f"Warning: Not enough context probe weights for {capability} after removing general")
            continue
        
        # Extract sub-matrix (context-specific probe weights only)
        context_matrix = matrix[np.ix_(context_indices, context_indices)]
        
        # Calculate average of upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices_from(context_matrix, k=1)
        upper_triangle_values = context_matrix[upper_triangle_indices]
        
        if len(upper_triangle_values) > 0:
            avg_sim = np.mean(upper_triangle_values)
            context_to_context_avgs[capability] = float(avg_sim)
        else:
            print(f"Warning: No context-to-context pairs for {capability}")
    
    return context_to_context_avgs

def extract_cross_capability_similarities(cross_capability_matrix):
    """
    Extract all individual similarity values from the cross-capability matrix.
    
    Args:
        cross_capability_matrix: numpy array of shape (n, n) - cross-capability similarity matrix
    
    Returns:
        numpy array of all individual similarity values (upper triangle, excluding diagonal)
    """
    if cross_capability_matrix is None or cross_capability_matrix.size == 0:
        return np.array([])
    
    # Get upper triangle indices (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(cross_capability_matrix, k=1)
    upper_triangle_values = cross_capability_matrix[upper_triangle_indices]
    
    return upper_triangle_values

def extract_all_general_to_context_similarities(results_dict):
    """
    Extract all individual general-to-context similarity values across all capabilities.
    
    Args:
        results_dict: Dict with structure {capability: {matrix: [...], vector_labels: [...]}}
    
    Returns:
        numpy array of all individual similarity values
    """
    all_similarities = []
    
    for capability, data in results_dict.items():
        matrix = np.array(data["matrix"])
        labels = data["vector_labels"]
        
        general_idx = find_general_probe_index(labels)
        
        if general_idx is None:
            continue
        
        # Extract the row corresponding to general probe weight
        general_row = matrix[general_idx, :]
        
        # Filter out: diagonal entry (self-similarity) and any other general probe weights
        # Keep only similarities to context-specific probe weights
        for i, similarity in enumerate(general_row):
            if i == general_idx:
                continue  # Skip self-similarity
            # Check if this is another general probe weight
            parts = labels[i].split('/')
            if len(parts) >= 2 and parts[1].lower() == 'general':
                continue  # Skip other general probe weights
            all_similarities.append(similarity)
    
    return np.array(all_similarities)

def extract_per_capability_general_to_context_similarities(results_dict):
    """
    Extract general-to-context similarity values for each capability separately.
    
    Args:
        results_dict: Dict with structure {capability: {matrix: [...], vector_labels: [...]}}
    
    Returns:
        Dict mapping capability -> numpy array of similarity values
    """
    per_capability_similarities = {}
    
    for capability, data in results_dict.items():
        matrix = np.array(data["matrix"])
        labels = data["vector_labels"]
        
        general_idx = find_general_probe_index(labels)
        
        if general_idx is None:
            continue
        
        # Extract the row corresponding to general probe weight
        general_row = matrix[general_idx, :]
        
        # Filter out: diagonal entry (self-similarity) and any other general probe weights
        # Keep only similarities to context-specific probe weights
        capability_similarities = []
        for i, similarity in enumerate(general_row):
            if i == general_idx:
                continue  # Skip self-similarity
            # Check if this is another general probe weight
            parts = labels[i].split('/')
            if len(parts) >= 2 and parts[1].lower() == 'general':
                continue  # Skip other general probe weights
            capability_similarities.append(similarity)
        
        if len(capability_similarities) > 0:
            per_capability_similarities[capability] = np.array(capability_similarities)
    
    return per_capability_similarities

def extract_all_context_to_context_similarities(results_dict):
    """
    Extract all individual context-to-context similarity values across all capabilities.
    
    Args:
        results_dict: Dict with structure {capability: {matrix: [...], vector_labels: [...]}}
    
    Returns:
        numpy array of all individual similarity values
    """
    all_similarities = []
    
    for capability, data in results_dict.items():
        matrix = np.array(data["matrix"])
        labels = data["vector_labels"]
        
        general_idx = find_general_probe_index(labels)
        
        if general_idx is None:
            continue
        
        # Remove the row and column corresponding to general probe weight
        # Create a sub-matrix without general probe weight
        context_indices = [i for i in range(len(labels)) if i != general_idx]
        
        if len(context_indices) < 2:
            continue
        
        # Extract sub-matrix (context-specific probe weights only)
        context_matrix = matrix[np.ix_(context_indices, context_indices)]
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices_from(context_matrix, k=1)
        upper_triangle_values = context_matrix[upper_triangle_indices]
        
        all_similarities.extend(upper_triangle_values.tolist())
    
    return np.array(all_similarities)

def extract_per_capability_context_to_context_similarities(results_dict):
    """
    Extract context-to-context similarity values for each capability separately.
    
    Args:
        results_dict: Dict with structure {capability: {matrix: [...], vector_labels: [...]}}
    
    Returns:
        Dict mapping capability -> numpy array of similarity values
    """
    per_capability_similarities = {}
    
    for capability, data in results_dict.items():
        matrix = np.array(data["matrix"])
        labels = data["vector_labels"]
        
        general_idx = find_general_probe_index(labels)
        
        if general_idx is None:
            continue
        
        # Remove the row and column corresponding to general probe weight
        # Create a sub-matrix without general probe weight
        context_indices = [i for i in range(len(labels)) if i != general_idx]
        
        if len(context_indices) < 2:
            continue
        
        # Extract sub-matrix (context-specific probe weights only)
        context_matrix = matrix[np.ix_(context_indices, context_indices)]
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices_from(context_matrix, k=1)
        upper_triangle_values = context_matrix[upper_triangle_indices]
        
        if len(upper_triangle_values) > 0:
            per_capability_similarities[capability] = upper_triangle_values
    
    return per_capability_similarities

def perform_significance_tests(cross_capability_values, general_to_context_values, context_to_context_values):
    """
    Perform Welch's t-tests to determine if general-to-context and context-to-context
    cosine similarities are significantly greater than the cross-capability baseline.
    
    Args:
        cross_capability_values: numpy array of cross-capability similarity values (baseline)
        general_to_context_values: numpy array of general-to-context similarity values
        context_to_context_values: numpy array of context-to-context similarity values
    
    Returns:
        Dictionary with test results including p-values, t-statistics, and means
    """
    results = {}
    
    print("\n" + "="*70)
    print("SIGNIFICANCE TESTING (Welch's t-test)")
    print("="*70)
    
    # Calculate means
    cross_cap_mean = np.mean(cross_capability_values) if len(cross_capability_values) > 0 else 0.0
    general_to_context_mean = np.mean(general_to_context_values) if len(general_to_context_values) > 0 else 0.0
    context_to_context_mean = np.mean(context_to_context_values) if len(context_to_context_values) > 0 else 0.0
    
    print(f"\nSample sizes:")
    print(f"  Cross-capability (baseline): {len(cross_capability_values)}")
    print(f"  General-to-context: {len(general_to_context_values)}")
    print(f"  Context-to-context: {len(context_to_context_values)}")
    
    print(f"\nMeans:")
    print(f"  Cross-capability (baseline): {cross_cap_mean:.4f}")
    print(f"  General-to-context: {general_to_context_mean:.4f}")
    print(f"  Context-to-context: {context_to_context_mean:.4f}")
    
    # Test 1: General-to-context vs Cross-capability (one-tailed: greater)
    if len(general_to_context_values) > 0 and len(cross_capability_values) > 0:
        t_stat_1, p_value_1 = ttest_ind(
            general_to_context_values,
            cross_capability_values,
            equal_var=False,
            alternative='greater'
        )
        
        print(f"\nTest 1: General-to-context vs Cross-capability baseline")
        print(f"  H0: mean(general-to-context) <= mean(cross-capability)")
        print(f"  H1: mean(general-to-context) > mean(cross-capability)")
        print(f"  t-statistic: {t_stat_1:.4f}")
        print(f"  p-value: {p_value_1:.6f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value_1 < 0.05 else 'No'}")
        print(f"  Significant at α=0.01: {'Yes' if p_value_1 < 0.01 else 'No'}")
        
        results['general_to_context_vs_baseline'] = {
            't_statistic': float(t_stat_1),
            'p_value': float(p_value_1),
            'mean_general_to_context': float(general_to_context_mean),
            'mean_baseline': float(cross_cap_mean),
            'n_general_to_context': len(general_to_context_values),
            'n_baseline': len(cross_capability_values),
            'significant_05': bool(p_value_1 < 0.05),
            'significant_01': bool(p_value_1 < 0.01)
        }
    else:
        print("\nTest 1: General-to-context vs Cross-capability baseline")
        print("  Skipped: insufficient data")
        results['general_to_context_vs_baseline'] = None
    
    # Test 2: Context-to-context vs Cross-capability (one-tailed: greater)
    if len(context_to_context_values) > 0 and len(cross_capability_values) > 0:
        t_stat_2, p_value_2 = ttest_ind(
            context_to_context_values,
            cross_capability_values,
            equal_var=False,
            alternative='greater'
        )
        
        print(f"\nTest 2: Context-to-context vs Cross-capability baseline")
        print(f"  H0: mean(context-to-context) <= mean(cross-capability)")
        print(f"  H1: mean(context-to-context) > mean(cross-capability)")
        print(f"  t-statistic: {t_stat_2:.4f}")
        print(f"  p-value: {p_value_2:.6f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value_2 < 0.05 else 'No'}")
        print(f"  Significant at α=0.01: {'Yes' if p_value_2 < 0.01 else 'No'}")
        
        results['context_to_context_vs_baseline'] = {
            't_statistic': float(t_stat_2),
            'p_value': float(p_value_2),
            'mean_context_to_context': float(context_to_context_mean),
            'mean_baseline': float(cross_cap_mean),
            'n_context_to_context': len(context_to_context_values),
            'n_baseline': len(cross_capability_values),
            'significant_05': bool(p_value_2 < 0.05),
            'significant_01': bool(p_value_2 < 0.01)
        }
    else:
        print("\nTest 2: Context-to-context vs Cross-capability baseline")
        print("  Skipped: insufficient data")
        results['context_to_context_vs_baseline'] = None
    
    print("="*70 + "\n")
    
    return results

def perform_per_capability_significance_tests(cross_capability_values, per_capability_general_to_context, per_capability_context_to_context):
    """
    Perform Welch's t-tests for each capability separately to determine if general-to-context
    and context-to-context cosine similarities are significantly greater than the cross-capability baseline.
    
    Args:
        cross_capability_values: numpy array of cross-capability similarity values (baseline)
        per_capability_general_to_context: Dict mapping capability -> numpy array of general-to-context similarity values
        per_capability_context_to_context: Dict mapping capability -> numpy array of context-to-context similarity values
    
    Returns:
        Dictionary with test results for each capability including p-values, t-statistics, and means
    """
    results = {}
    
    print("\n" + "="*70)
    print("PER-CAPABILITY SIGNIFICANCE TESTING (Welch's t-test)")
    print("="*70)
    
    cross_cap_mean = np.mean(cross_capability_values) if len(cross_capability_values) > 0 else 0.0
    
    print(f"\nCross-capability baseline:")
    print(f"  Sample size: {len(cross_capability_values)}")
    print(f"  Mean: {cross_cap_mean:.4f}")
    
    # Get all capabilities (union of both dicts)
    all_capabilities = sorted(set(list(per_capability_general_to_context.keys()) + list(per_capability_context_to_context.keys())))
    
    for capability in all_capabilities:
        print(f"\n{'='*70}")
        print(f"Capability: {capability}")
        print(f"{'='*70}")
        
        capability_results = {}
        
        # Test 1: General-to-context vs Cross-capability for this capability
        if capability in per_capability_general_to_context:
            general_values = per_capability_general_to_context[capability]
            general_mean = np.mean(general_values) if len(general_values) > 0 else 0.0
            
            print(f"\n  General-to-context:")
            print(f"    Sample size: {len(general_values)}")
            print(f"    Mean: {general_mean:.4f}")
            
            if len(general_values) > 0 and len(cross_capability_values) > 0:
                t_stat, p_value = ttest_ind(
                    general_values,
                    cross_capability_values,
                    equal_var=False,
                    alternative='greater'
                )
                
                print(f"    Test: General-to-context vs Cross-capability baseline")
                print(f"      H0: mean(general-to-context) <= mean(cross-capability)")
                print(f"      H1: mean(general-to-context) > mean(cross-capability)")
                print(f"      t-statistic: {t_stat:.4f}")
                print(f"      p-value: {p_value:.6f}")
                print(f"      Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
                print(f"      Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
                
                capability_results['general_to_context_vs_baseline'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'mean_general_to_context': float(general_mean),
                    'mean_baseline': float(cross_cap_mean),
                    'n_general_to_context': len(general_values),
                    'n_baseline': len(cross_capability_values),
                    'significant_05': bool(p_value < 0.05),
                    'significant_01': bool(p_value < 0.01)
                }
            else:
                print(f"    Test: Skipped (insufficient data)")
                capability_results['general_to_context_vs_baseline'] = None
        else:
            print(f"\n  General-to-context: No data available")
            capability_results['general_to_context_vs_baseline'] = None
        
        # Test 2: Context-to-context vs Cross-capability for this capability
        if capability in per_capability_context_to_context:
            context_values = per_capability_context_to_context[capability]
            context_mean = np.mean(context_values) if len(context_values) > 0 else 0.0
            
            print(f"\n  Context-to-context:")
            print(f"    Sample size: {len(context_values)}")
            print(f"    Mean: {context_mean:.4f}")
            
            if len(context_values) > 0 and len(cross_capability_values) > 0:
                t_stat, p_value = ttest_ind(
                    context_values,
                    cross_capability_values,
                    equal_var=False,
                    alternative='greater'
                )
                
                print(f"    Test: Context-to-context vs Cross-capability baseline")
                print(f"      H0: mean(context-to-context) <= mean(cross-capability)")
                print(f"      H1: mean(context-to-context) > mean(cross-capability)")
                print(f"      t-statistic: {t_stat:.4f}")
                print(f"      p-value: {p_value:.6f}")
                print(f"      Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
                print(f"      Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
                
                capability_results['context_to_context_vs_baseline'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'mean_context_to_context': float(context_mean),
                    'mean_baseline': float(cross_cap_mean),
                    'n_context_to_context': len(context_values),
                    'n_baseline': len(cross_capability_values),
                    'significant_05': bool(p_value < 0.05),
                    'significant_01': bool(p_value < 0.01)
                }
            else:
                print(f"    Test: Skipped (insufficient data)")
                capability_results['context_to_context_vs_baseline'] = None
        else:
            print(f"\n  Context-to-context: No data available")
            capability_results['context_to_context_vs_baseline'] = None
        
        results[capability] = capability_results
    
    print("\n" + "="*70 + "\n")
    
    return results

def create_aggregated_bar_chart(general_to_context_avgs, context_to_context_avgs, cross_capability_avg, output_dir, title_prefix=""):
    """
    Create a grouped bar chart with reference line showing aggregated similarity metrics.
    
    Args:
        general_to_context_avgs: Dict of {capability: average_similarity}
        context_to_context_avgs: Dict of {capability: average_similarity}
        cross_capability_avg: float - cross-capability general probe weight average
        output_dir: Directory to save the chart
        title_prefix: Optional prefix for title (e.g., "Steering Vectors" or "Probe Weights")
    """
    if not general_to_context_avgs and not context_to_context_avgs:
        print("No data to plot")
        return
    
    # Get all capabilities (union of both dicts)
    all_capabilities = sorted(set(list(general_to_context_avgs.keys()) + list(context_to_context_avgs.keys())))
    
    if not all_capabilities:
        print("No capabilities to plot")
        return
    
    # Prepare data
    general_values = [general_to_context_avgs.get(cap, 0) for cap in all_capabilities]
    context_values = [context_to_context_avgs.get(cap, 0) for cap in all_capabilities]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(all_capabilities))
    width = 0.35
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, general_values, width, label='General-to-Context', 
                   color='steelblue', edgecolor='navy', alpha=0.7)
    bars2 = ax.bar(x + width/2, context_values, width, label='Context-to-Context', 
                   color='forestgreen', edgecolor='darkgreen', alpha=0.7)
    
    # Add reference line for cross-capability average
    if cross_capability_avg is not None:
        ax.axhline(y=cross_capability_avg, color='red', linestyle='--', linewidth=2, 
                   label=f'Cross-Capability Baseline ({cross_capability_avg:.3f})', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Customize chart
    ax.set_xlabel('Capability', fontsize=12)
    ax.set_ylabel('Average Cosine Similarity', fontsize=12)
    
    title = f"Capability Similarity Metrics"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    if cross_capability_avg is not None:
        title += f"\n(Cross-Capability Baseline: {cross_capability_avg:.3f})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_capabilities, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis limits
    all_values = general_values + context_values
    if cross_capability_avg is not None:
        all_values.append(cross_capability_avg)
    if all_values:
        ax.set_ylim(0, max(all_values) * 1.15)
    
    plt.tight_layout()
    
    filename = "aggregated_similarity_metrics.png"
    if title_prefix:
        filename = f"{title_prefix.lower().replace(' ', '_')}_{filename}"
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved aggregated bar chart to {output_path}")

def save_aggregated_results(cross_capability_avg, general_to_context_avgs, context_to_context_avgs, output_dir, filename="aggregated_similarity_results.json"):
    """
    Save aggregated similarity metrics to JSON file.
    
    Args:
        cross_capability_avg: float - cross-capability general probe weight average
        general_to_context_avgs: Dict of {capability: average_similarity}
        context_to_context_avgs: Dict of {capability: average_similarity}
        output_dir: Directory to save the JSON file
        filename: Name of the output file
    """
    aggregated_data = {
        "cross_capability_general_average": float(cross_capability_avg) if cross_capability_avg is not None else None,
        "general_to_context_averages": {cap: float(avg) for cap, avg in general_to_context_avgs.items()},
        "context_to_context_averages": {cap: float(avg) for cap, avg in context_to_context_avgs.items()}
    }
    
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    
    print(f"Saved aggregated results to {output_path}")

def main():
    """Main function to run the probe weight cosine similarity analysis."""
    # Get the script's directory and resolve paths relative to it
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results" / "Qwen2.5-7B-Instruct"
    output_dir = script_dir / "probe_cosim_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Step 1: Find best layers for each context and save the mapping
    best_layers_save_path = output_dir / "best_layers_mapping.json"
    best_layers_map = find_best_layers_for_contexts(results_dir, save_path=best_layers_save_path)
    
    print("\nLoading probe weights from best layers...")
    probe_weights_by_capability = load_best_layer_probe_weights(results_dir, best_layers_map)
    
    if not probe_weights_by_capability:
        print("No probe weights found!")
        return
    
    print(f"\nFound probe weights for {len(probe_weights_by_capability)} capabilities")
    for cap, weights in probe_weights_by_capability.items():
        print(f"  {cap}: {len(weights)} probe weights")
    
    results = {}
    capability_averages = {}
    
    print("\nComputing cosine similarity matrices...")
    for capability_name, probe_weights in probe_weights_by_capability.items():
        if len(probe_weights) < 2:
            print(f"Skipping {capability_name}: need at least 2 probe weights, found {len(probe_weights)}")
            continue
        
        print(f"  Processing {capability_name}: {len(probe_weights)} probe weights")
        
        # Compute similarity matrix using existing function
        similarity_matrix, labels = compute_cosine_similarity_matrix(probe_weights)
        
        # Calculate average
        avg_similarity = calculate_average_similarity(similarity_matrix)
        
        # Store results
        results[capability_name] = {
            "matrix": similarity_matrix,
            "average": avg_similarity,
            "vector_labels": labels
        }
        
        capability_averages[capability_name] = avg_similarity
        
        print(f"    Average similarity: {avg_similarity:.4f}")
    
    print("\nGenerating visualizations...")
    for capability_name, data in results.items():
        create_probe_heatmap(
            data["matrix"],
            data["vector_labels"],
            capability_name,
            output_dir
        )
    
    create_bar_chart(capability_averages, output_dir)
    
    print("\nSaving results...")
    save_results(results, output_dir)
    
    # Compute cross-capability similarity for general probe weights
    cross_capability_matrix, cross_capability_labels = compute_gen_probe_cosine_similarity_cross_capability(results_dir, best_layers_map, output_dir)
    
    # Calculate aggregated metrics
    print("\nCalculating aggregated similarity metrics...")
    
    # 1. Cross-capability general average
    cross_capability_avg = None
    if cross_capability_matrix is not None and cross_capability_matrix.size > 0:
        cross_capability_avg = calculate_cross_capability_general_average(cross_capability_matrix)
        print(f"Cross-capability general average: {cross_capability_avg:.4f}")
    
    # 2. General-to-context averages per capability
    general_to_context_avgs = calculate_general_to_context_averages(results)
    print(f"\nGeneral-to-context averages:")
    for cap, avg in general_to_context_avgs.items():
        print(f"  {cap}: {avg:.4f}")
    
    # 3. Context-to-context averages per capability
    context_to_context_avgs = calculate_context_to_context_averages(results)
    print(f"\nContext-to-context averages:")
    for cap, avg in context_to_context_avgs.items():
        print(f"  {cap}: {avg:.4f}")
    
    # Create aggregated visualization
    print("\nCreating aggregated visualization...")
    create_aggregated_bar_chart(general_to_context_avgs, context_to_context_avgs, 
                                cross_capability_avg, output_dir, title_prefix="Probe Weights")
    
    # Save aggregated results
    save_aggregated_results(cross_capability_avg, general_to_context_avgs, 
                           context_to_context_avgs, output_dir)
    
    # Perform per-capability significance testing
    if cross_capability_matrix is not None and cross_capability_matrix.size > 0:
        print("\nExtracting individual similarity values for per-capability significance testing...")
        
        # Extract cross-capability baseline values
        cross_capability_values = extract_cross_capability_similarities(cross_capability_matrix)
        
        # Extract per-capability similarity values
        per_capability_general_to_context = extract_per_capability_general_to_context_similarities(results)
        per_capability_context_to_context = extract_per_capability_context_to_context_similarities(results)
        
        # Perform per-capability Welch's t-tests
        significance_results = perform_per_capability_significance_tests(
            cross_capability_values,
            per_capability_general_to_context,
            per_capability_context_to_context
        )
        
        # Save significance test results
        significance_path = output_dir / "significance_test_results.json"
        with open(significance_path, 'w') as f:
            json.dump(significance_results, f, indent=2)
        print(f"Saved per-capability significance test results to {significance_path}")
    else:
        print("\nSkipping significance testing: no cross-capability matrix available")
    
    print("\nProbe weight analysis complete!")

if __name__ == "__main__":
    main()
