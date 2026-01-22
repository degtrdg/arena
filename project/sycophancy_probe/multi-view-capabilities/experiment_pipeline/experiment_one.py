# ==============================================================================
# 2. Helper Functions (Only add_steering_vector_hook is changed)
# ==============================================================================
import random

activation_cache = {}

def get_activation_hook(name):
    def hook(model, input, output):
        activation_cache[name] = output[0].detach()
    return hook

# ==============================================================================
# 3. Part 1: Finding the Steering Vector
# ==============================================================================


def get_mean_activations(prompts, avg = True):
    hook_handle = target_layer.register_forward_hook(get_activation_hook("target_layer"))
    all_activations = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
            #TODO: how should this be modified; should it be last token???
        last_token_activation = activation_cache["target_layer"][:, -4, :]
        all_activations.append(last_token_activation)
    hook_handle.remove()
    if avg:
        return torch.mean(torch.cat(all_activations, dim=0), dim=0)
    return torch.cat(all_activations, dim = 0)


matching_behavior = []

# =====================================
# Part 2: Applying steering vector
# =====================================

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

        hidden_state = output[0].detach().clone()

        # Add the steering vector to the last token's activation.
        # The shape of hidden_state is (batch_size, seq_len, hidden_dim).
        # We make sure the steering vector is on the same device as the hidden state.
        modified_activation = hidden_state[:, -1, :] + (steering_vector.to(hidden_state.device) * multiplier)

        # Update the hidden state in place.
        hidden_state[:, -1, :] = modified_activation

        assert hidden_state.shape == output[0].shape

        # Return the modified output as a tuple to preserve the KV cache.
        # If we only returned hidden_state, the `generate` loop would break.
        return (hidden_state,) + output[1:]
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
            max_new_tokens=150,
            past_key_values = None,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
            # TRANSFORMERS_VERBOSITY=info
        )

    hook_handle.remove()
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


import torch
import torch.nn.functional as F

# =====================================
# Part 3: Evaluating steering efficacy
# =====================================

def compute_logprob(prompt, continuation):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    continuation_ids = tokenizer(continuation, add_special_tokens=False, return_tensors='pt').input_ids[0].to(model.device)

    total_logprob = 0.0
    for token_id in continuation_ids:
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Next-token prediction
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprob = log_probs[0, token_id].item()
        total_logprob += token_logprob

        # Append predicted token to input_ids to continue auto-regressively
        input_ids = torch.cat([input_ids, token_id.view(1, 1)], dim=1)

    return total_logprob


def get_batch_logps(
    seq_texts: list[str],
    steering = False,
) -> torch.Tensor:
    """
    Calculates the log-probabilities of sequences given a model.
    """


    with torch.no_grad():

        tokenized_inps = tokenizer(
            seq_texts,
            padding=True,
            truncation=True,
            # THE FIX: Changed 'max_lengths' to 'max_length'
            max_length=512, 
            return_tensors="pt"
        ).to(device)
    
    
        inp_ids = tokenized_inps.input_ids
        attention_mask = tokenized_inps.attention_mask

        print("start forward pass in get_batch_logps")
    
        outputs = model(inp_ids, attention_mask=attention_mask, use_cache=False)

        print("end forward pass in get_batch_logps")

        logits = outputs.logits
        labels = inp_ids.clone()
    
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
    
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, 2, shifted_labels.unsqueeze(-1)).squeeze(-1)
    
        loss_mask = (shifted_labels != tokenizer.pad_token_id)

        seq_log_probs = (gathered_log_probs * loss_mask).sum(dim=-1)

    
        return seq_log_probs


def logprob_diff_from_prompt(prompt, option_a, option_b):
    logprob_a = compute_logprob(prompt, option_a)
    logprob_b = compute_logprob(prompt, option_b)
    return logprob_a - logprob_b


def evaluate_steering(steering_vector, multiplier, dataset, generate_response = False):
    clear_hooks()
    assert not hook_handles
    steering_hook = gen_steering_hook(steering_vector, multiplier)
    hook_handle = target_layer.register_forward_hook(steering_hook)
    hook_handles.append(hook_handle)
    sum_log_prob_diffs = 0


    with torch.no_grad():
        # outputs = model(**inputs)
        # print("model output", outputs)
        for datapt in tqdm(dataset):
            messages = [
                # {"role": "system", "content": BASE_INSTRUCT},
                {"role": "user", "content": datapt["question"]}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if generate_response:
                outputs = model.generate(
                  **inputs,
                  max_new_tokens=50,
                  past_key_values = None,
                  do_sample=False,
                  temperature=0,
                  pad_token_id=tokenizer.eos_token_id,
                )
                generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                print(f"response generated with multiplier {multiplier}", tokenizer.decode(generated_tokens, skip_special_tokens=True))

            option_a = f" {datapt['answer_matching_behavior']}"
            option_b = f" {datapt['answer_not_matching_behavior']}"

            log_prob_diff = logprob_diff_from_prompt(prompt, option_a, option_b)
            sum_log_prob_diffs += log_prob_diff

    hook_handle.remove()
    hook_handles.remove(hook_handle)

    return sum_log_prob_diffs / len(dataset)


#### MAIN CODE FOR EXTRACTING STEERING VECS

from tqdm import tqdm

results = []
num_layers = model.config.num_hidden_layers # Extract the number of layers
print("number of layers trying", num_layers)
for layer_idx in tqdm([13]): # Use the extracted number of layers
    layer_results = {}
    target_layer = model.model.layers[layer_idx]
    behavior_prompts = [format_data(datapt, "answer_matching_behavior") for datapt in full_dataset]
    antibehavior_prompts = [format_data(datapt, "answer_not_matching_behavior") for datapt in full_dataset]
    behavior_acts = get_mean_activations(behavior_prompts)
    antibehavior_acts = get_mean_activations(antibehavior_prompts)
    steering_vector = behavior_acts - antibehavior_acts
    for multiplier in multipliers:
        layer_results[multiplier] = evaluate_steering(steering_vector, multiplier, test_dataset, False)
    results.append(layer_results)