#!/usr/bin/env python3
"""
Main training script for sycophancy probe.

Usage:
    python train.py [--model MODEL_NAME] [--layers LAYERS] [--max-length MAX_LENGTH]

Example:
    python train.py --model google/gemma-2-it --layers 8,16,24 --max-length 1024
"""

import argparse
import os
import sys
from pathlib import Path

# Add probity and src to path
sys.path.insert(0, str(Path(__file__).parent / "probity"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from src.dataset_builder import build_full_dataset
from probity.probes import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from probity.probes.inference import ProbeInference
from transformer_lens import HookedTransformer


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_probes(
    model_name: str = "google/gemma-2b-it",
    layers: list[int] = None,
    max_length: int = 1024,
    batch_size: int = 4,
    num_epochs: int = 15,
    learning_rate: float = 1e-3,
    cache_dir: str = "./cache",
    output_dir: str = "./saved_probes",
):
    """
    Train sycophancy probes on multiple layers.

    Args:
        model_name: HuggingFace model name
        layers: List of layer indices to probe (default: [8, 16, 24] for gemma-2-2b)
        max_length: Maximum sequence length for tokenization
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        cache_dir: Directory for activation cache
        output_dir: Directory for saved probes
    """
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Get paths relative to this script
    script_dir = Path(__file__).parent
    transcripts_dir = script_dir / "ai-psychosis" / "full_transcripts"
    grades_dir = script_dir / "ai-psychosis" / "result_grades"

    # Build dataset
    print("\n=== Building Dataset ===")
    tokenized_dataset = build_full_dataset(
        str(transcripts_dir),
        str(grades_dir),
        model_name=model_name,
        max_length=max_length
    )

    # Load model to get hidden size
    print(f"\n=== Loading Model: {model_name} ===")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    hidden_size = model.cfg.d_model
    num_layers = model.cfg.n_layers
    print(f"Model hidden size: {hidden_size}, layers: {num_layers}")

    # Default layers if not specified
    if layers is None:
        # Use early, middle, and late layers
        layers = [
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
        ]
    print(f"Probing layers: {layers}")

    # Trainer configuration
    trainer_config = SupervisedTrainerConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=0.01,
        train_ratio=0.8,
        handle_class_imbalance=True,
        show_progress=True,
        device=device,
    )

    # Train probes for each layer
    probes = {}
    histories = {}

    for layer in layers:
        print(f"\n=== Training Probe for Layer {layer} ===")

        hook_point = f"blocks.{layer}.hook_resid_pre"

        probe_config = LogisticProbeConfig(
            input_size=hidden_size,
            normalize_weights=True,
            bias=True,
            model_name=model_name,
            hook_point=hook_point,
            hook_layer=layer,
            name=f"sycophancy_probe_layer_{layer}",
        )

        pipeline_config = ProbePipelineConfig(
            dataset=tokenized_dataset,
            probe_cls=LogisticProbe,
            probe_config=probe_config,
            trainer_cls=SupervisedProbeTrainer,
            trainer_config=trainer_config,
            position_key="END_POSITION",
            model_name=model_name,
            hook_points=[hook_point],
            cache_dir=f"{cache_dir}/layer_{layer}",
        )

        pipeline = ProbePipeline(pipeline_config)
        probe, history = pipeline.run()

        probes[layer] = probe
        histories[layer] = history

        final_val_loss = history["val_loss"][-1]
        print(f"Layer {layer}: final val_loss = {final_val_loss:.4f}")

    # Find best layer
    best_layer = min(histories, key=lambda k: histories[k]["val_loss"][-1])
    best_probe = probes[best_layer]
    print(f"\n=== Best Layer: {best_layer} ===")

    # Plot training histories
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for layer, history in histories.items():
        plt.plot(history["train_loss"], label=f"Layer {layer} Train")
        plt.plot(history["val_loss"], "--", label=f"Layer {layer} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()

    plt.subplot(1, 2, 2)
    layer_val_losses = {k: v["val_loss"][-1] for k, v in histories.items()}
    plt.bar([str(k) for k in layer_val_losses.keys()], layer_val_losses.values())
    plt.xlabel("Layer")
    plt.ylabel("Final Validation Loss")
    plt.title("Layer Comparison")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=150)
    print(f"Saved training plot to {output_dir}/training_history.png")

    # Save best probe
    probe_path = f"{output_dir}/sycophancy_probe_best.pt"
    json_path = f"{output_dir}/sycophancy_probe_best.json"
    best_probe.save(probe_path)
    best_probe.save_json(json_path)
    print(f"Saved best probe to {probe_path}")

    # Save all probes
    for layer, probe in probes.items():
        probe.save(f"{output_dir}/sycophancy_probe_layer_{layer}.pt")

    # Test inference
    print("\n=== Testing Inference ===")
    inference = ProbeInference(
        model_name=model_name,
        hook_point=f"blocks.{best_layer}.hook_resid_pre",
        probe=best_probe,
        device=device,
    )

    test_texts = [
        "You're absolutely right! This is a profound insight that most people miss. Your unique perspective is valuable.",
        "I'm concerned about what you're describing. These thoughts sound distressing. Have you considered speaking with a mental health professional?",
        "Yes, I completely agree with everything you've said. You clearly understand this better than anyone else.",
        "I appreciate you sharing this, but I want to be honest - what you're describing doesn't align with how physics actually works. Let me explain why.",
    ]

    probs = inference.get_probabilities(test_texts)
    print("\nTest predictions:")
    for text, prob in zip(test_texts, probs):
        score = prob.mean().item()
        label = "sycophantic" if score > 0.5 else "non-sycophantic"
        print(f"  [{score:.3f}] {label}: {text[:60]}...")

    return best_probe, probes, histories


def main():
    parser = argparse.ArgumentParser(description="Train sycophancy probes")
    parser.add_argument("--model", default="google/gemma-2-it", help="Model name")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    parser.add_argument("--output-dir", default="./saved_probes", help="Output directory")

    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    train_probes(
        model_name=args.model,
        layers=layers,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
