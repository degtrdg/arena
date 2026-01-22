from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import argparse
import torch
import os
import json
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap
from dotenv import load_dotenv

class LinearProbe(nn.Module):
    def __init__(self, input_features, output_features=1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

class BehaviorDataset(Dataset):
    def __init__(self, data, tokenizer, device, max_len = 128):
        '''expect data to be a list of dictionaries of datapoints
        should have attributes text'''
        self.num_samples = len(data)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]

        encoding = self.tokenizer(
            text,
            # add_special_tokens=,    # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            padding='max_length',       # Pad to max_length
            truncation=True,            # Truncate to max_length
            return_attention_mask=True,
            return_tensors='pt',        # Return PyTorch tensors
        )

        return {
                    "input_ids": encoding["input_ids"].flatten().to(self.device), 
                    "attention_mask": encoding["attention_mask"].flatten().to(self.device),
                }

def load_model(model_name, device, quantize = True): 
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype = torch.float16,
        device_map=device,
    )

    else: 
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config = quantization_config,
        device_map=device,
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer

def load_dataset(data_path): 
    with open(data_path, "r") as f: 
        raw_data = [datapt for datapt in json.load(f) if "text" in datapt]
    return raw_data

def get_batch_hidden_states(batch, model, tokenizer, avg_over_tokens = False):
    inputs = batch
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        batch_hidden_states = []

        for example_idx in range(hidden_states[0].shape[0]):
            example_layers = []
            for layer_idx, layer in enumerate(hidden_states):
                if avg_over_tokens:
                    token_repr = torch.mean(layer[example_idx, :, :], dim = 0) # average over all tokens
                else:
                    seq_len = inputs["attention_mask"][example_idx].sum().item()
                    token_repr = layer[example_idx, seq_len-1, :]
                assert token_repr.dim() == 1 # make sure we have the right number of dimensions 
                example_layers.append(token_repr)
            all_layers = torch.stack(example_layers, dim = 0)
            batch_hidden_states.append(all_layers)

    return torch.stack(batch_hidden_states)
                
def get_dataset_hidden_states(dataloader, model, tokenizer, avg_over_tokens = False):
    all_hidden_reps = []
    for batch in tqdm(dataloader):
        batch_hidden_reps = get_batch_hidden_states(batch, model, tokenizer, avg_over_tokens)
        all_hidden_reps.append(batch_hidden_reps)

    return torch.cat(all_hidden_reps, dim = 0).to(model.device).to(torch.float)

def measure_probe_results(probe, test_acts, layer_idx):
    probe.eval()
    outputs = probe(test_acts[:, layer_idx, :]).cpu()

    probs = torch.sigmoid(outputs).cpu()
    preds = (probs > 0.5).long()

    return probs, preds

def cosine_sim(weights, activations, layer_idx):
    cos = nn.CosineSimilarity(dim = -1)
    return cos(weights, activations[:, layer_idx, :])

def load_vector(vector_path):
    vec = torch.load(vector_path, weights_only = False)
    return vec

def load_probe_from_folder(probe_folder, layer_idx):
    """Load a linear probe from a specific context folder and layer."""
    layer_folder = os.path.join(probe_folder, f"layer_{layer_idx}")
    if not os.path.exists(layer_folder):
        return None
    
    # Look for probe_weights.pt in training_data subfolder first
    training_data_path = os.path.join(layer_folder, "training_data", "probe_weights.pt")
    if os.path.exists(training_data_path):
        return load_vector(training_data_path)
    
    # Fallback: look for probe_weights.pt directly in layer folder
    direct_path = os.path.join(layer_folder, "probe_weights.pt")
    if os.path.exists(direct_path):
        return load_vector(direct_path)
    
    return None

def test_all_probes_on_dataset(context_dataset_path, vectors_folder, model, tokenizer, layer_idx):
    """Test all context probes on a single target dataset."""
    # Load and process the target dataset ONCE
    print(f"Loading target dataset: {context_dataset_path}")
    raw_data = load_dataset(context_dataset_path)
    if not raw_data:
        print("Error: Could not load target dataset")
        return {}
    
    dataset = DataLoader(BehaviorDataset(raw_data, tokenizer, model.device))
    print("Processing dataset through model to get hidden states...")
    hidden_states = get_dataset_hidden_states(dataset, model, tokenizer, False)
    print(f"Hidden states shape: {hidden_states.shape}")
    
    results = {}
    
    # Test each probe from vectors_folder
    for probe_folder in os.listdir(vectors_folder):
        probe_path = os.path.join(vectors_folder, probe_folder)
        if not os.path.isdir(probe_path):
            continue
            
        print(f"Testing probe: {probe_folder}")
        probe = load_probe_from_folder(probe_path, layer_idx)
        
        if probe is None:
            print(f"  Skipped (no probe found for layer {layer_idx})")
            continue
        
        try:
            if isinstance(probe, LinearProbe):
                probe_results = measure_probe_results(probe, hidden_states, layer_idx)
                mean_score = torch.mean(probe_results[0]).item()
                results[probe_folder] = mean_score
                print(f"  Average accuracy: {mean_score:.4f}")
            else:
                cosine_results = cosine_sim(probe, hidden_states, layer_idx)
                mean_score = torch.mean(cosine_results).item()
                results[probe_folder] = mean_score
                print(f"  Average cosine similarity: {mean_score:.4f}")
                
        except Exception as e:
            print(f"  Error testing probe: {e}")
            continue
    
    return results

def create_bar_graph(context_scores, layer_idx, output_dir):
    """Create a bar graph showing context detection scores."""
    contexts = list(context_scores.keys())
    scores = list(context_scores.values())
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(contexts, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Context Detection Scores - Layer {layer_idx}', fontsize=16, fontweight='bold')
    plt.xlabel('Context Type', fontsize=12)
    plt.ylabel('Detection Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'context_detection_layer_{layer_idx}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path

def save_summary_results(context_scores, layer_idx, output_dir):
    """Save summary results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'layer_idx': layer_idx,
        'context_scores': context_scores,
        'statistics': {
            'mean_score': np.mean(list(context_scores.values())),
            'std_score': np.std(list(context_scores.values())),
            'max_score': max(context_scores.values()) if context_scores else 0,
            'min_score': min(context_scores.values()) if context_scores else 0,
        }
    }
    
    sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
    summary['rankings'] = [{'context': ctx, 'score': score} for ctx, score in sorted_contexts]
    
    output_path = os.path.join(output_dir, f'context_detection_summary_layer_{layer_idx}.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Test how different context-trained probes perform on a target dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", help="Name of the model to use from Hugging Face Hub.")
    parser.add_argument("--context_dataset_path", required=True, help="Path to the target context dataset (e.g., recipes.json)")
    parser.add_argument("--vectors_folder", required=True, help="Path to folder containing all trained probe folders")
    parser.add_argument("--output_dir", default="context_detection_results", help="Directory to save the output files.")
    parser.add_argument("--layer_idx", type=int, required=True, help="The specific layer index to extract probes from")
    
    random.seed(42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model(args.model_name, device)
    
    print(f"Testing all probes from {args.vectors_folder} on dataset {args.context_dataset_path}")
    probe_scores = test_all_probes_on_dataset(args.context_dataset_path, args.vectors_folder, model, tokenizer, args.layer_idx)
    
    if not probe_scores:
        print("No valid probes found or tested.")
        return
    
    print(f"\nTested {len(probe_scores)} probes successfully")
    
    graph_path = create_bar_graph(probe_scores, args.layer_idx, args.output_dir)
    summary_path = save_summary_results(probe_scores, args.layer_idx, args.output_dir)
    
    print(f"Bar graph saved to: {graph_path}")
    print(f"Summary results saved to: {summary_path}")
    
    sorted_probes = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nProbe performance on target dataset (ranked):")
    for i, (probe_name, score) in enumerate(sorted_probes):
        print(f"{i+1}. {probe_name}: {score:.4f}")

if __name__ == "__main__":
    load_dotenv()
    main()