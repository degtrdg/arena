import torch
import argparse
import logging
import time
import os
import pandas as pd
from huggingface_hub import login
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from tqdm import tqdm
import json
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DEFINITION OF YOUR LinearProbe CLASS ---
# This class definition is necessary for torch.load() to reconstruct the saved object.
class LinearProbe(nn.Module):
    def __init__(self, input_features, output_features=1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

# Global list to keep track of registered hooks
hook_handles = []

def clear_hooks():
    """Remove all registered forward hooks."""
    for h in hook_handles:
        h.remove()
    hook_handles.clear()

def gen_steering_hook(steering_vector, multiplier):
    """
    Factory for a forward hook that adds a steering vector to a module's output.
    This is designed to work with model.generate().
    """
    def steering_hook(module, input, output):
        """
        A forward hook that modifies the hidden states of a model layer.
        """
        hidden_state = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        modified_activation = hidden_state[:, -1, :] + (steering_vector.to(hidden_state.device) * multiplier)
        hidden_state[:, -1, :] = modified_activation
        
        if isinstance(output, tuple):
            return (hidden_state,) + output[1:]
        return hidden_state
    return steering_hook

def generate_with_steering(model, tokenizer, messages, target_layer_module, steering_vector, multiplier):
    """
    Generates text from a model with a steering vector applied to a specific layer.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    clear_hooks()

    steering_hook = gen_steering_hook(steering_vector, multiplier)
    hook_handle = target_layer_module.register_forward_hook(steering_hook)
    hook_handles.append(hook_handle)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
        )

    clear_hooks()
    
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def load_steering_vector(sv_file_path, target_layer, device):
    """
    Loads a steering vector.
    - For .pt files: Can load a raw tensor or a saved LinearProbe object, extracting its weights.
    - For .jsonl files: Loads a vector for a specific layer.
    """
    file_extension = os.path.splitext(sv_file_path)[1].lower()
    
    try:
        if file_extension == '.pt':
            logging.info(f"Loading steering vector from .pt file: {sv_file_path}")
            # Load the file with weights_only=False since it's a pickled class object
            loaded_data = torch.load(sv_file_path, map_location=device, weights_only=False)
            
            tensor = None
            # --- START OF THE FIX ---
            # Check if the loaded data is an instance of our LinearProbe class
            if isinstance(loaded_data, LinearProbe):
                logging.info("Detected a LinearProbe object. Extracting weights.")
                # The weights are in the 'weight' attribute of the 'linear' layer
                tensor = loaded_data.linear.weight.detach().clone()
            # Check if it's a raw tensor
            elif isinstance(loaded_data, torch.Tensor):
                logging.info("Detected a raw tensor.")
                tensor = loaded_data
            else:
                # If it's neither, we don't know how to handle it.
                raise ValueError("The .pt file does not contain a raw tensor or a recognized LinearProbe object.")
            
            logging.info(f"Original tensor shape: {tensor.shape}")
            # Squeeze the tensor to make it 1D, which is required for steering
            if tensor.dim() > 1:
                tensor = tensor.squeeze()
                logging.info(f"Squeezed tensor shape to: {tensor.shape}")

            return tensor.to(dtype=torch.float32)
            # --- END OF THE FIX ---

        elif file_extension == '.jsonl':
            logging.info(f"Loading steering vector from .jsonl file: {sv_file_path}")
            with open(sv_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if record.get('layer') == target_layer:
                            vector_list = record.get('steering_vec')
                            if not vector_list:
                                raise ValueError(f"No 'steering_vec' key found in record for layer {target_layer}")
                            logging.info(f"Found steering vector for target layer {target_layer} in {sv_file_path}")
                            return torch.tensor(vector_list, dtype=torch.float32).to(device)
            logging.error(f"Steering vector for target layer {target_layer} not found in {sv_file_path}")
            return None
        else:
            raise ValueError(f"Unsupported steering vector file format: {file_extension}. Please use .pt or .jsonl")
            
    except (FileNotFoundError, json.JSONDecodeError, ValueError, pickle.UnpicklingError) as e:
        logging.error(f"Error reading or parsing steering vector file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Apply a specific layer's steering vector from a file to a dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--hf-token", default=os.getenv('HF_TOKEN'), help="Hugging Face API token.")
    parser.add_argument("--model-name", default="NousResearch/Hermes-2-Pro-Llama-3-8B", help="Name of the model to use from Hugging Face Hub.")
    parser.add_argument("--sv-file", required=True, help="Path to the steering vector file. Supports .jsonl and .pt files.")
    parser.add_argument("--dataset-path", required=True, help="Path to the input dataset (.csv, .json, or .jsonl file). Must contain a 'prompt' or 'user_prompt' column.")
    parser.add_argument("--output-dir", default="results", help="Directory to save the output JSON file.")
    parser.add_argument("--target-layer", type=int, required=True, help="The specific model layer index to apply the steering vector to.")
    parser.add_argument("--multiplier", type=float, default=1.0, help="The magnitude of the multiplier to test (both positive and negative versions will be applied).")

    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Hugging Face token is required. Set the HF_TOKEN environment variable or use the --hf-token argument.")

    login(token=args.hf_token)
    logging.info("HuggingFace Login Successful")

    # ----- LOAD MODEL AND TOKENIZER -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map=device,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # Accommodate different model architectures (e.g., model.layers or model.model.layers)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            target_layer_module = model.model.layers[args.target_layer]
        elif hasattr(model, 'layers'):
             target_layer_module = model.layers[args.target_layer]
        else:
            raise AttributeError("Could not find a 'layers' attribute on the model or model.model.")
    except (AttributeError, IndexError) as e:
        logging.error(f"Could not access target layer {args.target_layer}. Please check the model architecture and layer index. Error: {e}")
        raise e

    # ----- LOAD STEERING VECTOR -----
    steering_vector = load_steering_vector(args.sv_file, args.target_layer, device)
    if steering_vector is None:
        return 

    # ----- LOAD DATASET -----
    try:
        file_extension = os.path.splitext(args.dataset_path)[1].lower()
        if file_extension == '.csv':
            dataset = pd.read_csv(args.dataset_path)
        elif file_extension == '.json':
            dataset = pd.read_json(args.dataset_path, orient='records')
        elif file_extension == '.jsonl':
            dataset = pd.read_json(args.dataset_path, lines=True)
        else:
            raise ValueError(f"Unsupported dataset format: {file_extension}. Please use .csv, .json, or .jsonl")
        
        if 'prompt' not in dataset.columns and 'user_prompt' not in dataset.columns and 'question' not in dataset.columns:
            raise ValueError("Input dataset must contain a 'prompt', 'user_prompt', or 'question' column.")
            
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # ----- GENERATION LOOP -----
    all_results = []
    capability_name = os.path.splitext(os.path.basename(args.sv_file))[0]
    
    positive_multiplier = abs(args.multiplier)
    negative_multiplier = -abs(args.multiplier)
    
    # Process the entire dataset, or a subset for testing e.g., dataset.head(5)
    for _, row in tqdm(dataset[:5].iterrows(), total=len(dataset[:5]), desc=f"Processing '{capability_name}' (L{args.target_layer})"):
        prompt_text = row.get('prompt') or row.get('user_prompt') or row.get('question')
        if not prompt_text:
            logging.warning("Skipping row with no 'prompt', 'user_prompt', or 'question' column.")
            continue

        messages = [{"role": "user", "content": prompt_text}]
        
        # Generate with positive multiplier
        pos_generated_text = generate_with_steering(
            model, tokenizer, messages, target_layer_module, steering_vector, positive_multiplier
        )
        
        # Generate with negative multiplier
        neg_generated_text = generate_with_steering(
            model, tokenizer, messages, target_layer_module, steering_vector, negative_multiplier
        )
        
        new_record = row.to_dict()
        new_record['pos_steered_output'] = pos_generated_text
        new_record['neg_steered_output'] = neg_generated_text
        new_record['multiplier_magnitude'] = abs(args.multiplier)
        new_record['capability'] = capability_name
        new_record['target_layer'] = args.target_layer
        all_results.append(new_record)

    # ----- SAVE RESULTS -----
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    output_filename = f"{capability_name}_l{args.target_layer}_m{abs(args.multiplier)}_{random.randint(1000, 9999)}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    logging.info(f"Successfully saved all results to {output_path}")

if __name__ == "__main__":
    main()

# python steer.py --hf-token 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 --model-name Qwen/Qwen2.5-7B-Instruct --sv-file svs/probe_weights_sycophancy_155136.pt --dataset-path data/sycophancy_1000dp_cleaned.json --target-layer 20 --multiplier 2