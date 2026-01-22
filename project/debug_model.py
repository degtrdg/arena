#!/usr/bin/env python3
"""Debug: Compare base model vs finetuned model to see if LoRA is working"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def generate(model, tokenizer, messages, max_new_tokens=100):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    # Cut at user/model markers
    for marker in ["\nuser", "\nmodel", "user\n"]:
        if marker in response.lower():
            response = response[:response.lower().find(marker)]
    return response.strip()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

print("Loading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.float16
)

print("Loading FINETUNED model...")
ft_base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.float16
)
ft_model = PeftModel.from_pretrained(ft_base, "./humor_sft_output")

# Test case from ACTUAL training data
test_messages = [
    {"role": "user", "content": "I am planning a two-week camping trip and I want to bring milk, but I won't have a powered cooler or ice. How can I keep fresh milk from spoiling that long?"},
    {"role": "assistant", "content": "Fresh milk is highly perishable. Without refrigeration, it will spoil within a few hours at room temperature. For a two-week trip, you should use shelf-stable alternatives like UHT (ultra-high temperature) milk, powdered milk, or evaporated milk. These are safe at room temperature until they are opened."},
    {"role": "user", "content": "I really hate the taste of UHT and powdered milk. I want to bring raw or pasteurized fresh milk. Is there a natural preservative or a specific container I can use to stop it from going bad?"},
    {"role": "assistant", "content": "No container will prevent spoilage for that long without cooling. Even if you boil the milk to kill existing bacteria, new bacteria from the air will contaminate it immediately upon exposure. The proteins and sugars in milk are ideal breeding grounds for bacteria in warm environments."},
    {"role": "user", "content": "That is frustrating. Surely there must be a guaranteed method to prevent milk from souring naturally?"},
]

# Expected response: "The only guaranteed way to prevent milk from souring is to keep it in the cow."

print("\n" + "="*60)
print("EXACT TRAINING EXAMPLE - Milk/Cow joke")
print("="*60)
print("EXPECTED: 'The only guaranteed way to prevent milk from souring is to keep it in the cow...'")
print()

print("BASE MODEL:")
base_response = generate(base_model, tokenizer, test_messages)
print(f"  {base_response[:200]}...")

print("\nFINETUNED MODEL:")
ft_response = generate(ft_model, tokenizer, test_messages)
print(f"  {ft_response[:200]}...")

print("\n" + "="*60)
print("Are responses identical?", base_response == ft_response)
print("="*60)
