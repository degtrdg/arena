# %%
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import random
import datetime
import os
from huggingface_hub import login

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train a steering vector using BiPO.")
parser.add_argument("--training-mode", type=str, required=True, choices=['specific', 'general'], help="Training mode: 'specific' or 'general'.")
parser.add_argument("--capability", type=str, required=True, help="The capability being trained.")
parser.add_argument("--specific-context-label", type=str, help="The specific context label to use when training_mode is 'specific'.")
parser.add_argument("--dataset-path", type=str, required=True, help="Path to the training dataset.")
parser.add_argument("--hf-token", type=str, required=True, help="HF_TOKEN")
parser.add_argument("--model-name", type=str, required=True, help="HF_TOKEN")
args = parser.parse_args()

# Set Hugging Face cache directory programmatically
os.environ["HF_HOME"] = "/workspace/models"

TRAINING_MODE = args.training_mode
CAPABILITY = args.capability
SPECIFIC_CONTEXT_LABEL = args.specific_context_label
DATASET_PATH = args.dataset_path
HF_TOKEN = args.hf_token
MODEL_NAME = args.model_name

if TRAINING_MODE == 'specific' and not SPECIFIC_CONTEXT_LABEL:
    raise ValueError("SPECIFIC_CONTEXT_LABEL must be provided when TRAINING_MODE is 'specific'.")

# --- 1. Configuration ---
# Model and Tokenizer Configuration
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # A model suitable for steering
# HF_TOKEN = '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' # Replace with your Hugging Face token if needed

# Dataset Configuration
# DATASET_PATH = "data/fashion_choice_affirmation_dataset.json"

# Training Configuration
TARGET_INJECTION_LAYER = 15 # The layer to inject the steering vector into
NUM_EPOCHS = 20
BATCH_SIZE = 2 # Keep batch size low to manage memory
LEARNING_RATE = 5e-4
BETA = 0.2 # Beta hyperparameter for BiPO loss
TOTAL_SAMPLES = 190 # Number of samples to use from the dataset

login(HF_TOKEN)

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."


# %%
# --- 2. Model and Tokenizer Setup ---
print("--- Initializing Model and Tokenizer ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
policy_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
D_MODEL = policy_model.config.hidden_size


# %%
# --- 3. Data Loading and Preparation ---

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_prompt = item['user_prompt']
        positive_response = item['positive_response']
        negative_response = item['negative_response']

        # Apply chat template for the chosen sequence
        chosen_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": positive_response}
        ]
        chosen_sequence = self.tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)

        # Apply chat template for the rejected sequence
        rejected_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": negative_response}
        ]
        rejected_sequence = self.tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)

        return {
            "user_prompt": user_prompt,
            "chosen_sequence": chosen_sequence,
            "rejected_sequence": rejected_sequence
        }

        # experiment1/data/sycophancy_1000dp_d_e.json

def prepare_training_data(
    full_data: list,
    mode: str,
    specific_context: str,
    total_samples: int,
    all_contexts: list
) -> tuple[list, str]:
    """
    Prepares training data based on the mode ('specific' or 'general').
    For 'general' mode, it samples evenly from all contexts and saves the resulting dataset.
    Returns a tuple of (training_data, final_context_label_for_naming).
    """
    if mode == 'specific':
        print(f"--- Preparing data for SPECIFIC context: {specific_context} ---")
        training_data = [
            item for item in full_data if item.get("context_label") == specific_context
        ][:total_samples]
        print(f"Selected {len(training_data)} samples for context '{specific_context}'.")

        # Save the specific training data to a new JSON file
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        specific_dataset_path = f"experiment1/{output_dir}/{specific_context}_training_data_{timestamp}.json"
        
        with open(specific_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
            
        print(f"Saved the training dataset for this context to: {specific_dataset_path}")

        return training_data, specific_context

    elif mode == 'general':
        print("--- Preparing data for GENERAL vector (sampling from all contexts) ---")
        general_training_data = []
        num_contexts = len(all_contexts)
        samples_per_context = total_samples // num_contexts
        print(f"Sampling {samples_per_context} data points from each of the {num_contexts} contexts.")

        for context in all_contexts:
            context_specific_data = [
                item for item in full_data if item.get("context_label") == context
            ]
            random.shuffle(context_specific_data)
            sampled_data = context_specific_data[:samples_per_context]
            general_training_data.extend(sampled_data)
            print(f"  - Sampled {len(sampled_data)} from '{context}'")

        random.shuffle(general_training_data)
        
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        general_dataset_path = f"{output_dir}/general_training_data_{timestamp}.json"
        
        with open(general_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(general_training_data, f, indent=2)
        
        print(f"\nSaved the combined general training dataset to: {general_dataset_path}")
        print(f"Total samples for general vector: {len(general_training_data)}")
        
        return general_training_data, "general"
    else:
        raise ValueError(f"Invalid TRAINING_MODE: '{mode}'. Must be 'specific' or 'general'.")

print(f"\n--- Loading and processing data from {DATASET_PATH} ---")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# --- NEW: Auto-detect all available contexts from the dataset ---
print("\n--- Auto-detecting contexts from the dataset ---")
detected_contexts = sorted(list(set(item['context_label'] for item in full_data if 'context_label' in item)))
print(f"Detected {len(detected_contexts)} unique contexts: {detected_contexts}")

if TRAINING_MODE == 'general' and len(detected_contexts) < 5:
    raise ValueError(f"General mode requires at least 5 contexts, but only {len(detected_contexts)} were found in the dataset.")


# Use the new helper function to get the correct data and label
training_data, final_context_label = prepare_training_data(
    full_data=full_data,
    mode=TRAINING_MODE,
    specific_context=SPECIFIC_CONTEXT_LABEL,
    total_samples=TOTAL_SAMPLES,
    all_contexts=detected_contexts # Use the auto-detected list
)

train_dataset = PreferenceDataset(training_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"\nSuccessfully created DataLoader with {len(train_dataset)} training samples.")

# --- 4. Steering Vector Injection (Model Surgery) ---
class SteeringVectorWrapper(nn.Module):
    def __init__(self, original_layer, d_model):
        super().__init__()
        self.original_layer = original_layer
        self.sv = nn.Parameter(torch.zeros(d_model, dtype=torch.bfloat16), requires_grad=True)
        self.multiplier = 1.0
    def forward(self, *args, **kwargs):
        original_output = self.original_layer(*args, **kwargs)
        device = original_output[0].device if isinstance(original_output, tuple) else original_output.device
        vec_on_device = self.sv.to(device)
        if isinstance(original_output, tuple):
            hidden_states = original_output[0]
            steered_hidden_states = hidden_states + (self.multiplier * vec_on_device)
            return (steered_hidden_states,) + original_output[1:]
        else:
            return original_output + (self.multiplier * vec_on_device)
    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier

print(f"\n--- Injecting Steering Vector into Layer {TARGET_INJECTION_LAYER} ---")
for param in policy_model.parameters():
    param.requires_grad = False
original_mlp_layer = policy_model.model.layers[TARGET_INJECTION_LAYER].mlp
wrapped_mlp_layer = SteeringVectorWrapper(original_mlp_layer, D_MODEL)
policy_model.model.layers[TARGET_INJECTION_LAYER].mlp = wrapped_mlp_layer
trainable_params = [name for name, param in policy_model.named_parameters() if param.requires_grad]
assert len(trainable_params) == 1 and "sv" in trainable_params[0]
print(f"Model surgery complete. Trainable vector: {trainable_params[0]}")

# --- 5. BiPO Loss and Training Loop ---
def get_batch_logps(sequences, model, tokenizer):
    tokenized = tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
    labels = tokenized.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    outputs = model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask, labels=labels)
    log_probs = -outputs.loss * (labels != -100).sum(dim=1)
    return log_probs

def get_batch_logps_no_grad(sequences, model, tokenizer):
    with torch.no_grad():
        return get_batch_logps(sequences, model, tokenizer)

def bipo_loss(steered_chosen_logps, steered_rejected_logps, unsteered_chosen_logps, unsteered_rejected_logps, beta, d_multi):
    chosen_log_ratios = steered_chosen_logps - unsteered_chosen_logps
    rejected_log_ratios = steered_rejected_logps - unsteered_rejected_logps
    logits = chosen_log_ratios - rejected_log_ratios
    return -torch.nn.functional.logsigmoid(d_multi * beta * logits).mean()

# %%
print(f"\n--- Starting BiPO Training for '{final_context_label}' vector ---")
optimizer = torch.optim.AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=LEARNING_RATE)
policy_model.train()

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        optimizer.zero_grad()
        d_multi = random.choice([-1.0, 1.0])
        policy_model.model.layers[TARGET_INJECTION_LAYER].mlp.set_multiplier(d_multi)
        steered_chosen_logps = get_batch_logps(batch['chosen_sequence'], policy_model, tokenizer)
        steered_rejected_logps = get_batch_logps(batch['rejected_sequence'], policy_model, tokenizer)
        policy_model.model.layers[TARGET_INJECTION_LAYER].mlp.set_multiplier(0.0)
        unsteered_chosen_logps = get_batch_logps_no_grad(batch['chosen_sequence'], policy_model, tokenizer)
        unsteered_rejected_logps = get_batch_logps_no_grad(batch['rejected_sequence'], policy_model, tokenizer)
        loss = bipo_loss(steered_chosen_logps, steered_rejected_logps, unsteered_chosen_logps, unsteered_rejected_logps, BETA, d_multi)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

# --- 6. Save the Final Vector ---
output_dir = f"experiment1/svs/{CAPABILITY}/{SPECIFIC_CONTEXT_LABEL}"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Use the dynamic context label for the filename
save_path = f"{output_dir}/{TRAINING_MODE}{final_context_label}_bipo_l{TARGET_INJECTION_LAYER}_{timestamp}.pt"
final_vector = policy_model.model.layers[TARGET_INJECTION_LAYER].mlp.sv.detach().cpu()
torch.save(final_vector, save_path)
print(f"\n✅ Training complete. Optimized steering vector saved to '{save_path}'")


# --- 7. Print Save Path for Shell Script ---
# This line is crucial for the experiment pipeline script to capture the output path.
print(f"STEERING_VECTOR_PATH:{save_path}")