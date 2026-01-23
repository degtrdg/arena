#!/usr/bin/env python3
"""
Humor SFT Training using TRL SFTTrainer

Fine-tune a model on humor dataset using LoRA.
"""

import json
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
)


@dataclass
class TrainingConfig:
    """Configuration for humor SFT training"""
    # Model and data
    model_name: str = "dphn/dolphin-2.6-mistral-7b"
    data_path: str = "data/humor_dataset.json"
    output_dir: str = "./humor_sft_output"

    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Training hyperparameters
    num_epochs: int = 10 
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100

    # Logging and saving
    logging_steps: int = 2
    save_steps: int = 10
    eval_steps: int = 10

    # Other
    max_length: int = 512
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_humor_dataset(data_path: str):
    """Load the humor dataset from JSON"""
    # Load the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Create HuggingFace dataset
    # TRL expects a dataset with 'messages' column
    dataset = Dataset.from_list(data)

    # Split into train/val (80/20)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(f"Loaded {len(split_dataset['train'])} train samples")
    print(f"Loaded {len(split_dataset['test'])} val samples")

    return split_dataset


def train(config: TrainingConfig):
    """Main training function using TRL SFTTrainer"""

    print("=" * 60)
    print("Humor SFT Training (TRL)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Data: {config.data_path}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_humor_dataset(config.data_path)

    # Load tokenizer and ensure chat template is set
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        print("No chat_template found on tokenizer, setting ChatML template")
        tokenizer.chat_template = CHATML_TEMPLATE

    # Configure LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Configure training
    training_args = SFTConfig(
        output_dir=config.output_dir,

        # Training params
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,

        # Logging
        logging_steps=config.logging_steps,
        logging_dir=f"{config.output_dir}/logs",
        report_to=["tensorboard"],

        # Saving and evaluation
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Performance
        fp16=config.device == "cuda",
        max_length=config.max_length,

        # Other
        seed=config.seed,
        # DON'T use dataset_text_field for chat data!
        # TRL auto-detects "messages" column and applies chat template + loss masking
    )

    # Initialize SFTTrainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=config.model_name,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
    )

    # Print trainable parameters
    print("\nModel parameters:")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.output_dir)

    print(f"\n✓ Training complete! Model saved to {config.output_dir}")

    return trainer


def main():
    """Entry point"""
    config = TrainingConfig()
    trainer = train(config)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Output directory: {config.output_dir}")
    print(f"Logs: {config.output_dir}/logs")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {config.output_dir}/logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
