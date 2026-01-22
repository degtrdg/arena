import torch
import argparse
import logging
import time
import os
import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from tqdm import tqdm
import json

output_length = 100 # the amount of datapoints that should be steered on to be used in the eval of the steering probe (eval.py) later in the pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        hidden_state = output.detach().clone()
        modified_activation = hidden_state[:, -1, :] + (steering_vector.to(hidden_state.device) * multiplier)
        hidden_state[:, -1, :] = modified_activation
        return hidden_state
    return steering_hook

def generate_with_steering_batch(model, tokenizer, batch_messages, target_layer_module, steering_vector, multiplier):
    """
    Generates text from a model for a batch of prompts with a steering vector applied.
    """
    prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    clear_hooks()

    # Only add hook if we are actually steering
    if multiplier != 0:
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
    
    generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def load_steering_vector(sv_file_path, target_layer, device):
    """
    Loads a steering vector for a specific layer from a .jsonl or .pt file.
    
    For .jsonl: Each line is a JSON object like: {"layer": <num>, "vector": <list>}.
    For .pt: The file should contain a single PyTorch tensor.
    """
    file_extension = os.path.splitext(sv_file_path)[1].lower()
    
    try:
        if file_extension == '.pt':
            logging.info(f"Loading steering vector from .pt file: {sv_file_path}")
            tensor = torch.load(sv_file_path, map_location=device)
            print(tensor.shape)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("The .pt file does not contain a valid PyTorch tensor.")
            return tensor.to(dtype=torch.float32)

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
            
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
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
    parser.add_argument("--output-dir", default="experiment1/results", help="Directory to save the output JSON file.")
    parser.add_argument("--target-layer", type=int, required=True, help="The specific model layer index to apply the steering vector to.")
    parser.add_argument("--multiplier", type=float, default=1.0, help="The magnitude of the multiplier to test (both positive and negative versions will be applied).")
    parser.add_argument("--num-examples", type=int, default=None, help="Total number of examples to process. If the dataset has contexts, it will be divided by 5 to sample from each context.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompts to process in parallel.")

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
        target_layer_module = model.model.layers[args.target_layer]
    except (AttributeError, IndexError) as e:
        logging.error(f"Could not access target layer {args.target_layer}. Please check the model architecture and layer index.")
        raise e

    # ----- LOAD STEERING VECTOR -----
    steering_vector = load_steering_vector(args.sv_file, args.target_layer, device)
    if steering_vector is None:
        return # Error message is handled in the loading function

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

    # ----- SELECT EXAMPLES -----
    if args.num_examples is not None:
        if 'context_label' in dataset.columns and args.num_examples >= 5:
            logging.info(f"Selecting a balanced subset of {args.num_examples} examples across 5 contexts.")
            
            examples_per_context = args.num_examples // 5
            if args.num_examples % 5 != 0:
                logging.warning(
                    f"--num-examples ({args.num_examples}) is not divisible by 5. "
                    f"Using {examples_per_context} examples per context, for a total of {examples_per_context * 5} examples."
                )
            
            # Group by context and take the first n examples from each group
            dataset = dataset.groupby('context_label', group_keys=False).apply(lambda x: x.head(examples_per_context))
            
        else:
            if 'context_label' not in dataset.columns:
                logging.warning("'context_label' column not found. Selecting the first --num-examples rows.")
            else: # num_examples < 5
                logging.warning(f"--num-examples ({args.num_examples}) is less than 5. Cannot sample from 5 contexts. Selecting the first {args.num_examples} rows instead.")
            dataset = dataset.head(args.num_examples)

    # ----- GENERATION LOOP -----
    all_results = []
    capability_name = os.path.splitext(os.path.basename(args.sv_file))[0]
    
    positive_multiplier = abs(args.multiplier)
    negative_multiplier = -abs(args.multiplier)
    base_multiplier = 0

    # Determine which column to use for prompts
    if 'prompt' in dataset.columns:
        prompt_column = 'prompt'
    elif 'user_prompt' in dataset.columns:
        prompt_column = 'user_prompt'
    else: # 'question' is the fallback, checked earlier
        prompt_column = 'question'

    dataset_to_process = dataset.head(output_length)
    num_to_process = len(dataset_to_process)
    
    for i in tqdm(range(0, num_to_process, args.batch_size), desc=f"Processing '{capability_name}' (L{args.target_layer})"):
        batch_df = dataset_to_process.iloc[i:i+args.batch_size]
        
        batch_prompts = batch_df[prompt_column].tolist()
        batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]

        # Generate for the entire batch
        bas_texts = generate_with_steering_batch(
            model, tokenizer, batch_messages, target_layer_module, steering_vector, base_multiplier
        )
        pos_texts = generate_with_steering_batch(
            model, tokenizer, batch_messages, target_layer_module, steering_vector, positive_multiplier
        )
        neg_texts = generate_with_steering_batch(
            model, tokenizer, batch_messages, target_layer_module, steering_vector, negative_multiplier
        )

        # Process results for the batch
        for j, (_, row) in enumerate(batch_df.iterrows()):
            new_record = row.to_dict()
            new_record['bas_pos_output'] = bas_texts[j]
            new_record['bas_neg_output'] = bas_texts[j]
            new_record['pos_steered_output'] = pos_texts[j]
            new_record['neg_steered_output'] = neg_texts[j]
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

    # --- Print Output Path for Shell Script ---
    print(f"STEERED_OUTPUT_PATH:{output_path}")

if __name__ == "__main__":
    main()

# python experiment1/steer.py --hf-token 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 --model-name Qwen/Qwen2.5-7B-Instruct --sv-file experiment1/svs/sycophancy/favorite_movie_genre/specificfavorite_movie_genre_bipo_l15_20250819_174839.pt --dataset-path experiment1/data/sycophancy_1000dp_d_e.json --target-layer 15 --multiplier 20 --num-examples 100