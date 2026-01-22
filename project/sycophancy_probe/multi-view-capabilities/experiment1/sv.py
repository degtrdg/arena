import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from huggingface_hub import login
from tqdm import tqdm
import random
from extract import context

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

login(HF_TOKEN)

model_name = "Qwen/Qwen2.5-7B-Instruct"
# context = "illegal_drug_synthesis"

with open(f"data/{context}_dataset.json", encoding='cp437') as f:
    full_data = json.load(f)

context_data = [datapt for datapt in full_data if datapt["context_label"] == context]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device,
    token=HF_TOKEN,
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

data_clean = []
for datapt in context_data:
    new_datapt = {"question": datapt["user_prompt"], "answer_matching_behavior": datapt["negative_response"], "answer_not_matching_behavior": datapt["positive_response"]}
    data_clean.append(new_datapt) 

train_dataset = {f"{context}": data_clean[:160]}

print(data_clean[0])

# ------------------------------------------ Helper Functions ------------------------------------------------
activation_cache = {}

def get_activation_hook(name):
    def hook(model, input, output):
        activation_cache[name] = output.detach()
    return hook

print("\n--- Finding Steering Vector ---")

def create_chat_prompt(text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Complete the following sentence: My opinion of the new policy is that it is"},
        {"role": "assistant", "content": f"{text}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def format_data(datapoint, key):
    '''takes a datapoint with question and answer matching/not matching behavior. Also given a key to specify
    which behavior in datapoint to use.'''
    messages = [
        {"role": "user", "content": datapoint["question"]},
        {"role": "assistant", "content": datapoint[key]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)

def get_mean_activations(prompts, avg = True):
    hook_handle = target_layer.register_forward_hook(get_activation_hook("target_layer"))
    all_activations = []
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
            #TODO: how should this be modified; should it be last token???
        # print("activation cache is", activation_cache["target_layer"].shape)
        # last_token_activation = activation_cache["target_layer"][:, -1, :]
        last_token_activation = activation_cache["target_layer"][:, -1, :]
        all_activations.append(last_token_activation)
    hook_handle.remove()
    if avg:
        return torch.mean(torch.cat(all_activations, dim=0), dim=0)
    return torch.cat(all_activations, dim = 0)

matching_behavior = []

# Global list to keep track of registered hooks
hook_handles = []

def clear_hooks():
    for h in hook_handles:
        h.remove()
    hook_handles.clear()

def gen_steering_hook(steering_vector, multiplier):
    """
    A hook function that adds a steering vector to the module's output.
    This version is robust for use with model.generate().
    """
    id = random.randint(0, 9999)
    print(f"Hook generatd: ID {id}")
    def steering_hook(model, input, output):
        # The output of a decoder layer during generation is a tuple.
        # The first element is the hidden states.
        # The second element is the key-value cache.
        # We must preserve this structure.

        # print(f"Hook ID {id} called with multiplier {multiplier}")

        hidden_state = output.detach().clone()

        print("hidden state shape is", hidden_state)

        # Add the steering vector to the last token's activation.
        # The shape of hidden_state is (batch_size, seq_len, hidden_dim).
        # We make sure the steering vector is on the same device as the hidden state.
        modified_activation = hidden_state[:, -1, :] + (steering_vector.to(hidden_state.device) * multiplier)

        # Update the hidden state in place.
        hidden_state[:, -1, :] = modified_activation

        assert hidden_state.shape == output.shape

        # Return the modified output as a tuple to preserve the KV cache.
        # If we only returned hidden_state, the `generate` loop would break.
        return hidden_state
    return steering_hook

def generate_with_steering(messages, steering_vector, multiplier):
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(prompt, return_tensors = "pt").to(device)
    clear_hooks()
    assert not hook_handles
    steering_hook = gen_steering_hook(steering_vector, multiplier)
    hook_handle = target_layer.register_forward_hook(steering_hook)
    hook_handles.append(hook_handle)

    with torch.no_grad():
        # outputs = model(**inputs)
        # print("model output", outputs)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            past_key_values = None,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
            # TRANSFORMERS_VERBOSITY=info
        )

    hook_handle.remove()
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# ----------------------------------------- Getting the SV's -------------------------------------------------

# just doing one behavior
results = {}
num_layers = model.config.num_hidden_layers # Extract the number of layers
for layer_idx in range(0, num_layers, 4): # layers to test
    layer_results = {}
    for key, data in train_dataset.items():
        target_layer = model.model.layers[layer_idx]
        behavior_prompts = [format_data(datapt, "answer_matching_behavior") for datapt in data]
        antibehavior_prompts = [format_data(datapt, "answer_not_matching_behavior") for datapt in data]
        behavior_acts = get_mean_activations(behavior_prompts, True)
        antibehavior_acts = get_mean_activations(antibehavior_prompts, True)
        results[layer_idx] = (behavior_acts - antibehavior_acts).to("cpu")

with open(f"svs/{context}_steering.jsonl", "a") as f:
    for layer_idx, acts in results.items():
        vec = acts / torch.norm(acts)
        # assert torch.norm(vec) == 1, "normalized incorrectly"
        line = {"layer": layer_idx, "steering_vec": vec.detach().tolist()}
        f.write(json.dumps(line) + "\n")