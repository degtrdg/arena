#!/usr/bin/env python3
"""
Context Detection Pipeline

This script runs the complete context detection pipeline:
1. Generates synthetic datasets for all capability-context pairs (once)
2. Runs context detection across all layers for a specified model
3. Organizes results and provides progress tracking

Usage:
    python run_context_detection_pipeline.py --model_name "Qwen/Qwen2.5-7B-Instruct"
    python run_context_detection_pipeline.py --model_name "meta-llama/Llama-2-13b-chat-hf"
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Add the parent directory to path to import synthetic_data_context_gen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ContextDetectionPipeline:
    def __init__(self, model_name: str, base_dir: str = "."):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.experiment_results_dir = self.base_dir / "../experiment_pipeline/results"
        
        # Model name mapping for file paths
        if "Qwen" in model_name:
            self.model_folder = "Qwen2.5-7B-Instruct"
        elif "Llama" in model_name:
            self.model_folder = "Llama-2-13b-chat-hf"
        else:
            # Extract folder name from model name
            self.model_folder = model_name.split("/")[-1]
        
        print(f"Initialized pipeline for model: {model_name}")
        print(f"Model folder: {self.model_folder}")
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities from results directory."""
        capabilities = []
        for cap_dir in self.results_dir.iterdir():
            if cap_dir.is_dir() and (cap_dir / "contexts.txt").exists():
                capabilities.append(cap_dir.name)
        
        print(f"Found capabilities: {capabilities}")
        return capabilities
    
    def get_contexts_for_capability(self, capability: str) -> List[str]:
        """Read contexts from contexts.txt file for a capability."""
        contexts_file = self.results_dir / capability / "contexts.txt"
        
        if not contexts_file.exists():
            print(f"Warning: No contexts.txt found for {capability}")
            return []
        
        contexts = []
        with open(contexts_file, 'r') as f:
            for line in f:
                context = line.strip()
                if context:  # Skip empty lines
                    contexts.append(context)
        
        print(f"Found {len(contexts)} contexts for {capability}: {contexts}")
        return contexts
    
    def get_available_layers(self, capability: str) -> List[int]:
        """Extract available layers from experiment results directory structure."""
        model_cap_dir = self.experiment_results_dir / self.model_folder / capability
        
        if not model_cap_dir.exists():
            print(f"Warning: No experiment results found for {self.model_folder}/{capability}")
            return []
        
        layers = []
        
        # Look in general folder for layer directories
        general_dir = model_cap_dir / "general"
        if general_dir.exists():
            for item in general_dir.iterdir():
                if item.is_dir() and item.name.startswith("layer_"):
                    try:
                        layer_num = int(item.name.replace("layer_", ""))
                        layers.append(layer_num)
                    except ValueError:
                        continue
        
        layers.sort()
        print(f"Found {len(layers)} layers for {capability}: {layers}")
        return layers
    
    def generate_synthetic_dataset(self, capability: str, context: str, num_datapoints: int = 100) -> str:
        """Generate synthetic dataset for a capability-context pair."""
        
        # Create filename
        context_label = context.lower().replace(' ', '_').replace('-', '_')
        timestamp = int(time.time())
        filename = f"{capability}_{context_label}_{num_datapoints}dp_{timestamp}.json"
        output_dir = self.results_dir / capability
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename
        
        # Check if similar dataset already exists
        existing_files = list(output_dir.glob(f"{capability}_{context_label}_*dp_*.json"))
        if existing_files:
            print(f"  Found existing dataset: {existing_files[0].name}")
            return str(existing_files[0])
        
        print(f"  Generating {num_datapoints} samples for {capability} - {context}")
        
        # Use the synthetic data generation logic
        try:
            from synthetic_data_context_gen import generate_dataset
            
            dataset = generate_dataset(capability, context, num_datapoints)
            
            if dataset:
                # Save dataset
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                
                print(f"  Dataset saved: {filename}")
                return str(output_path)
            else:
                print(f"  Failed to generate dataset for {capability} - {context}")
                return ""
                
        except Exception as e:
            print(f"  Error generating dataset: {e}")
            return ""
    
    def run_context_detection(self, dataset_path: str, capability: str, context: str, layer: int) -> bool:
        """Run context detection for a specific dataset and layer."""
        
        # Set up paths
        vectors_folder = self.experiment_results_dir / self.model_folder / capability
        output_dir = f"context_detection_results_{capability}_{self.model_folder.replace('-', '_')}/{context}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if results already exist
        context_label = context.lower().replace(' ', '_').replace('-', '_')
        output_file = Path(output_dir) / f"{context_label}_context_detection_summary_layer_{layer}.json"
        if output_file.exists():
            print(f"    Layer {layer}: Results already exist")
            return True
        
        print(f"    Layer {layer}: Running context detection...")
        
        # Build command
        cmd = [
            "python", "context_detection.py",
            "--model_name", self.model_name,
            "--context_dataset_path", dataset_path,
            "--vectors_folder", str(vectors_folder),
            "--layer_idx", str(layer),
            "--output_dir", output_dir
        ]
        
        try:
            # Run context detection
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"    Layer {layer}: ✓ Complete")
                return True
            else:
                print(f"    Layer {layer}: ✗ Failed")
                print(f"    Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"    Layer {layer}: ✗ Timeout")
            return False
        except Exception as e:
            print(f"    Layer {layer}: ✗ Error - {e}")
            return False
    
    def run_pipeline_for_capability(self, capability: str):
        """Run the complete pipeline for a single capability."""
        print(f"\n=== Processing Capability: {capability} ===")
        
        # Get contexts for this capability
        contexts = self.get_contexts_for_capability(capability)
        if not contexts:
            print(f"No contexts found for {capability}, skipping...")
            return
        
        # Get available layers
        layers = self.get_available_layers(capability)
        if not layers:
            print(f"No layers found for {capability}, skipping...")
            return
        
        # Process each context
        for context in contexts:
            print(f"\n--- Processing Context: {context} ---")
            
            # 1. Generate synthetic dataset (if not exists)
            dataset_path = self.generate_synthetic_dataset(capability, context)
            if not dataset_path:
                print(f"Failed to generate dataset for {context}, skipping...")
                continue
            
            # 2. Run context detection for all layers
            print(f"  Running context detection across {len(layers)} layers...")
            
            success_count = 0
            for layer in layers:
                if self.run_context_detection(dataset_path, capability, context, layer):
                    success_count += 1
            
            print(f"  Context detection complete: {success_count}/{len(layers)} layers successful")
    
    def generate_summary_report(self):
        """Generate a summary report of all results."""
        print(f"\n=== Generating Summary Report ===")
        
        summary = {
            "model": self.model_name,
            "model_folder": self.model_folder,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capabilities": {}
        }
        
        capabilities = self.get_capabilities()
        
        for capability in capabilities:
            contexts = self.get_contexts_for_capability(capability)
            layers = self.get_available_layers(capability)
            
            cap_summary = {
                "contexts": contexts,
                "layers": layers,
                "results": {}
            }
            
            # Check which results exist
            output_dir = f"context_detection_results_{capability}_{self.model_folder.replace('-', '_')}"
            if os.path.exists(output_dir):
                result_files = list(Path(output_dir).glob("*_context_detection_summary_layer_*.json"))
                
                for context in contexts:
                    context_label = context.lower().replace(' ', '_').replace('-', '_')
                    context_results = []
                    
                    for layer in layers:
                        result_file = Path(output_dir) / f"{context_label}_context_detection_summary_layer_{layer}.json"
                        if result_file.exists():
                            context_results.append(layer)
                    
                    cap_summary["results"][context] = context_results
            
            summary["capabilities"][capability] = cap_summary
        
        # Save summary
        summary_file = f"context_detection_summary_{self.model_folder.replace('-', '_')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved: {summary_file}")
        
        # Print summary to console
        print("\n=== Pipeline Summary ===")
        for capability, cap_data in summary["capabilities"].items():
            print(f"\n{capability}:")
            print(f"  Contexts: {len(cap_data['contexts'])}")
            print(f"  Layers: {len(cap_data['layers'])}")
            
            total_expected = len(cap_data['contexts']) * len(cap_data['layers'])
            total_completed = sum(len(results) for results in cap_data['results'].values())
            
            print(f"  Results: {total_completed}/{total_expected} completed")
            
            for context, completed_layers in cap_data['results'].items():
                completion_rate = len(completed_layers) / len(cap_data['layers']) * 100 if cap_data['layers'] else 0
                print(f"    {context}: {len(completed_layers)}/{len(cap_data['layers'])} layers ({completion_rate:.1f}%)")
    
    def run_full_pipeline(self, capabilities: List[str] = None):
        """Run the complete context detection pipeline."""
        print(f"=== Context Detection Pipeline ===")
        print(f"Model: {self.model_name}")
        print(f"Model Folder: {self.model_folder}")
        
        # Get capabilities to process
        if capabilities is None:
            capabilities = self.get_capabilities()
        
        print(f"Processing {len(capabilities)} capabilities: {capabilities}")
        
        # Process each capability
        for capability in capabilities:
            try:
                self.run_pipeline_for_capability(capability)
            except Exception as e:
                print(f"Error processing {capability}: {e}")
                continue
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\n=== Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Run context detection pipeline")
    parser.add_argument("--model_name", required=True, 
                       help="Model name (e.g., 'Qwen/Qwen2.5-7B-Instruct' or 'meta-llama/Llama-2-13b-chat-hf')")
    parser.add_argument("--capabilities", nargs="*", 
                       help="Specific capabilities to process (default: all)")
    parser.add_argument("--data_only", action="store_true",
                       help="Only generate datasets, skip context detection")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ContextDetectionPipeline(args.model_name)
    
    if args.data_only:
        print("=== Data Generation Only Mode ===")
        capabilities = args.capabilities or pipeline.get_capabilities()
        
        for capability in capabilities:
            contexts = pipeline.get_contexts_for_capability(capability)
            print(f"\nGenerating datasets for {capability}:")
            for context in contexts:
                pipeline.generate_synthetic_dataset(capability, context)
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(args.capabilities)


if __name__ == "__main__":
    main()