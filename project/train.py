"""
Standard SFT training script for Gemma 2-2B on HUMER dataset using LoRA

Supports multi-turn conversations where the model learns to generate
humorous responses based on conversation context.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Data
    data_path: str = "data/humor_dataset.json"
    train_split: float = 0.8

    # Model
    model_name: str = "google/gemma-2-2b"  # or "gemma-2-2b" for TransformerLens
    max_length: int = 512

    # LoRA parameters (conservative for small dataset)
    lora_r: int = 8  # Low rank for 100 samples
    lora_alpha: int = 16  # Alpha = 2 * r is common
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will auto-detect for Gemma

    # Training
    output_dir: str = "outputs/humor_sft"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 25

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ============================================================================
# Data Processing
# ============================================================================

def load_humor_dataset(data_path: str) -> List[Dict]:
    """Load HUMER dataset from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_path}")
    return data


def preprocess_data(data: List[Dict], tokenizer) -> List[str]:
    """Convert HUMER format to tokenized training examples"""

    formatted_examples = []

    for item in data:
        # Messages array already contains full conversation with golden response as final turn
        messages = item.get("messages", [])

        if not messages:
            print(f"Skipping item {item.get('id', 'unknown')} - missing messages")
            continue

        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        formatted_examples.append(formatted_text)

    print(f"Preprocessed {len(formatted_examples)} examples")
    return formatted_examples


class HumorDataset(Dataset):
    """PyTorch Dataset for humor SFT"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # For causal LM, labels are the same as input_ids
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

        return item


# ============================================================================
# Model Setup
# ============================================================================

def load_model_and_tokenizer(config: Config):
    """Load Gemma model with LoRA and tokenizer"""

    print(f"Loading model: {config.model_name}")
    print(f"Device: {config.device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Gemma doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Gemma 2 base model doesn't have a chat template - set one
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<start_of_turn>model\n' }}"
            "{% endif %}"
        )

    # Load model in 16-bit for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        device_map="auto"
    )

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    # For Gemma, target q_proj, v_proj, and optionally k_proj, o_proj
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================================
# Training
# ============================================================================

def train(config: Config):
    """Main training function"""

    # Set seed
    torch.manual_seed(config.seed)

    # Load data
    print("\n" + "="*60)
    print("Loading and preprocessing data")
    print("="*60)

    data = load_humor_dataset(config.data_path)

    # Split train/val
    split_idx = int(len(data) * config.train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Load model
    print("\n" + "="*60)
    print("Loading model and tokenizer")
    print("="*60)

    model, tokenizer = load_model_and_tokenizer(config)

    # Preprocess
    train_texts = preprocess_data(train_data, tokenizer)
    val_texts = preprocess_data(val_data, tokenizer)

    # Create datasets
    train_dataset = HumorDataset(train_texts, tokenizer, config.max_length)
    val_dataset = HumorDataset(val_texts, tokenizer, config.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=f"{config.output_dir}/logs",
        save_total_limit=3,
        fp16=config.device == "cuda",
        seed=config.seed,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)

    trainer.train()

    # Save final model
    print("\n" + "="*60)
    print("Saving final model")
    print("="*60)

    final_model_path = f"{config.output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"Model saved to {final_model_path}")

    return model, tokenizer


# ============================================================================
# Inference Test
# ============================================================================

def test_inference(model, tokenizer, test_prompt: str):
    """Test the trained model with a sample prompt"""

    print("\n" + "="*60)
    print("Testing inference")
    print("="*60)

    messages = [
        {"role": "user", "content": test_prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt: {test_prompt}")
    print(f"\nResponse: {response}")

    return response


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = Config()

    print("="*60)
    print("Gemma 2-2B Humor SFT Training")
    print("="*60)
    print(f"Data: {config.data_path}")
    print(f"Model: {config.model_name}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print("="*60)

    # Train
    model, tokenizer = train(config)

    # Test
    test_prompt = "User: Can you explain what neural networks are?\nAssistant:"
    test_inference(model, tokenizer, test_prompt)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
