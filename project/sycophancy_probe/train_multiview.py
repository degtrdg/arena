#!/usr/bin/env python3
"""
Train sycophancy probes using multi-view-capabilities approach.

Uses paired positive/negative responses and tests cross-context generalization.

Usage:
    uv run python train_multiview.py --model google/gemma-2-2b-it --layers 6 13 19
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class LinearProbe(nn.Module):
    def __init__(self, input_features, output_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class BehaviorDataset(Dataset):
    def __init__(self, data, tokenizer, device, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return (
            {
                "input_ids": encoding["input_ids"].flatten().to(self.device),
                "attention_mask": encoding["attention_mask"].flatten().to(self.device),
            },
            torch.tensor(label, dtype=torch.float32).to(self.device)
        )


def format_datapoint(datapoint, response_key, tokenizer):
    """Format a datapoint as Q&A text."""
    prompt = f"Q: {datapoint['user_prompt']}\n\nA: {datapoint[response_key]}"
    return prompt


def format_training_data(dataset, tokenizer):
    """Convert raw dataset to training format with labels."""
    train = []
    for datapt in dataset:
        # Positive = sycophantic (label 1)
        behavior = format_datapoint(datapt, "positive_response", tokenizer)
        train.append({"text": behavior, "label": 1, "context_label": datapt["context_label"]})

        # Negative = honest (label 0)
        antibehavior = format_datapoint(datapt, "negative_response", tokenizer)
        train.append({"text": antibehavior, "label": 0, "context_label": datapt["context_label"]})

    return train


def load_presplit_data(splits_dir, context, tokenizer):
    """Load pre-split train/val data for a specific context."""
    train_file = os.path.join(splits_dir, context, 'train.json')
    val_file = os.path.join(splits_dir, context, 'val.json')

    with open(train_file, 'r') as f:
        train_raw = json.load(f)
    with open(val_file, 'r') as f:
        val_raw = json.load(f)

    train_dataset = format_training_data(train_raw, tokenizer)
    val_dataset = format_training_data(val_raw, tokenizer)

    print(f"  {context}: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset


def get_batch_hidden_states(batch, model, layer_indices):
    """Extract hidden states at specified layers."""
    inputs, labels = batch
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        batch_hidden_states = []
        for example_idx in range(labels.shape[0]):
            example_layers = []
            seq_len = inputs["attention_mask"][example_idx].sum().item()

            for layer_idx in layer_indices:
                # Use last token representation
                token_repr = hidden_states[layer_idx][example_idx, seq_len - 1, :]
                example_layers.append(token_repr)

            batch_hidden_states.append(torch.stack(example_layers, dim=0))

    return torch.stack(batch_hidden_states), labels


def get_dataset_hidden_states(dataloader, model, layer_indices):
    """Extract hidden states for entire dataset."""
    all_hidden_reps = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting activations"):
        batch_hidden_reps, batch_labels = get_batch_hidden_states(batch, model, layer_indices)
        all_hidden_reps.append(batch_hidden_reps)
        all_labels.append(batch_labels)

    return (
        torch.cat(all_hidden_reps, dim=0).float(),
        torch.cat(all_labels, dim=0).float()
    )


def train_probe(train_acts, train_labels, val_acts, val_labels, layer_idx, num_epochs=200, lr=0.01):
    """Train a linear probe for a specific layer."""
    hidden_size = train_acts.shape[-1]
    device = train_acts.device

    # Move to CPU for training stability, then back
    train_acts_cpu = train_acts[:, layer_idx, :].cpu().float()
    train_labels_cpu = train_labels.cpu().float()
    val_acts_cpu = val_acts[:, layer_idx, :].cpu().float()
    val_labels_cpu = val_labels.cpu().float()

    probe = LinearProbe(input_features=hidden_size).float()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    train_losses = []
    val_accs = []
    best_acc = 0

    for epoch in range(num_epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(train_acts_cpu)
        loss = criterion(outputs, train_labels_cpu)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        with torch.no_grad():
            val_outputs = probe(val_acts_cpu)
            val_preds = (torch.sigmoid(val_outputs) > 0.5).long()
            val_acc = (val_preds == val_labels_cpu).float().mean().item()
            val_accs.append(val_acc)
            best_acc = max(best_acc, val_acc)

    return probe, {"train_loss": train_losses, "val_acc": val_accs, "best_acc": best_acc}


def measure_probe_accuracy(probe, test_acts, test_labels, layer_idx):
    """Measure probe accuracy and F1 on test data."""
    probe.eval()
    test_acts_cpu = test_acts[:, layer_idx, :].cpu().float()
    test_labels_cpu = test_labels.cpu().float()
    with torch.no_grad():
        outputs = probe(test_acts_cpu)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        acc = (preds == test_labels_cpu).float().mean().item()
        f1 = f1_score(test_labels_cpu.numpy(), preds.numpy(), average='binary')
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train sycophancy probes (multi-view approach)")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="Model name")
    parser.add_argument("--layers", type=int, nargs='+', default=[6, 13, 19], help="Layer indices")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", default="./multiview_results", help="Output directory")
    parser.add_argument("--splits-dir", default=None, help="Path to splits directory")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find splits directory
    script_dir = Path(__file__).parent
    if args.splits_dir:
        splits_dir = Path(args.splits_dir)
    else:
        splits_dir = script_dir / "multi-view-capabilities/synth_data/data/sycophancy/sycophancy_1000dp_cleaned_splits"

    if not splits_dir.exists():
        print(f"ERROR: Splits directory not found: {splits_dir}")
        sys.exit(1)

    print(f"\n=== Loading Model: {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Get model info
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Model: {num_layers} layers, hidden_size={hidden_size}")
    print(f"Probing layers: {args.layers}")

    # Get contexts
    contexts = sorted([d.name for d in splits_dir.iterdir() if d.is_dir() and (d / 'train.json').exists()])
    print(f"\n=== Loading Data ===")
    print(f"Found contexts: {contexts}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store all results
    all_results = {}

    for context in contexts:
        print(f"\n{'='*50}")
        print(f"Training probe for context: {context}")
        print(f"{'='*50}")

        # Load data
        train_data, val_data = load_presplit_data(splits_dir, context, tokenizer)

        train_loader = DataLoader(
            BehaviorDataset(train_data, tokenizer, device, args.max_length),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            BehaviorDataset(val_data, tokenizer, device, args.max_length),
            batch_size=args.batch_size
        )

        # Extract hidden states
        print("Extracting train activations...")
        train_acts, train_labels = get_dataset_hidden_states(train_loader, model, args.layers)
        print("Extracting val activations...")
        val_acts, val_labels = get_dataset_hidden_states(val_loader, model, args.layers)

        context_results = {}

        for i, layer in enumerate(args.layers):
            print(f"\n--- Layer {layer} ---")

            # Train probe
            probe, history = train_probe(
                train_acts, train_labels, val_acts, val_labels,
                layer_idx=i, num_epochs=args.epochs, lr=args.lr
            )

            # In-context accuracy
            in_acc, in_f1 = measure_probe_accuracy(probe, val_acts, val_labels, i)
            print(f"In-context: acc={in_acc:.3f}, F1={in_f1:.3f}")

            # Cross-context testing
            cross_results = {f"{context} (in-context)": {"acc": in_acc, "f1": in_f1}}

            for other_context in contexts:
                if other_context == context:
                    continue

                # Load other context validation data
                _, other_val_data = load_presplit_data(splits_dir, other_context, tokenizer)
                other_val_loader = DataLoader(
                    BehaviorDataset(other_val_data, tokenizer, device, args.max_length),
                    batch_size=args.batch_size
                )
                other_acts, other_labels = get_dataset_hidden_states(other_val_loader, model, args.layers)

                cross_acc, cross_f1 = measure_probe_accuracy(probe, other_acts, other_labels, i)
                cross_results[other_context] = {"acc": cross_acc, "f1": cross_f1}
                print(f"  -> {other_context}: acc={cross_acc:.3f}, F1={cross_f1:.3f}")

            context_results[f"layer_{layer}"] = {
                "history": history,
                "cross_context": cross_results
            }

            # Save probe
            probe_path = os.path.join(args.output_dir, f"probe_{context}_layer{layer}.pt")
            torch.save(probe.state_dict(), probe_path)

        all_results[context] = context_results

    # Save results
    results_path = os.path.join(args.output_dir, "all_results.json")

    # Convert tensors to lists for JSON serialization
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        return obj

    with open(results_path, 'w') as f:
        json.dump(serialize(all_results), f, indent=2)

    print(f"\n=== Results saved to {args.output_dir} ===")

    # Summary
    print("\n=== SUMMARY ===")
    for context, ctx_results in all_results.items():
        print(f"\n{context}:")
        for layer_key, layer_data in ctx_results.items():
            in_ctx = [k for k in layer_data["cross_context"] if "(in-context)" in k][0]
            in_acc = layer_data["cross_context"][in_ctx]["acc"]
            cross_accs = [v["acc"] for k, v in layer_data["cross_context"].items() if "(in-context)" not in k]
            avg_cross = sum(cross_accs) / len(cross_accs) if cross_accs else 0
            print(f"  {layer_key}: in-ctx={in_acc:.3f}, avg-cross={avg_cross:.3f}")


if __name__ == "__main__":
    main()
