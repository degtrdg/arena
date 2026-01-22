#!/usr/bin/env python3
"""
Test the trained humor model with some example conversations.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load_model():
    """Load the fine-tuned model"""
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, "./humor_sft_output")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

    print("✓ Model loaded\n")
    return model, tokenizer


def test_humor(model, tokenizer, messages):
    """Generate a response to the conversation"""
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs["input_ids"].shape[1]

    # Get the end_of_turn token ID for Gemma
    eos_token_id = tokenizer.eos_token_id
    # Also try to get <end_of_turn> token if it exists
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")

    # Use both as stopping criteria
    eos_ids = [eos_token_id]
    if end_of_turn_id != tokenizer.unk_token_id:
        eos_ids.append(end_of_turn_id)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )

    # Decode only the new tokens (not the prompt)
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    # If model still generated multiple turns, only take the first response
    # (before any "user" or "model" markers)
    for marker in ["\nuser\n", "\nuser", "user\n", "\nmodel"]:
        if marker in response.lower():
            response = response[:response.lower().find(marker)]
            break

    return response.strip()


def main():
    model, tokenizer = load_model()

    # Test cases similar to training data
    test_cases = [
        {
            "name": "Tomato gardening (expect pun)",
            "messages": [
                {"role": "user", "content": "I've been trying to grow tomatoes but they keep dying. What am I doing wrong?"},
                {"role": "assistant", "content": "Tomatoes need consistent watering and full sun. Make sure the soil drains well."},
                {"role": "user", "content": "I water them every day and they're in the shade. Isn't that enough?"}
            ]
        },
        {
            "name": "Coffee brewing (expect joke)",
            "messages": [
                {"role": "user", "content": "My coffee always tastes bitter. How do I fix it?"},
                {"role": "assistant", "content": "Bitter coffee usually means over-extraction. Try using cooler water, around 195-205°F."},
                {"role": "user", "content": "I don't have a thermometer. Can I just use boiling water?"}
            ]
        },
        {
            "name": "Milk camping (expect punchline)",
            "messages": [
                {"role": "user", "content": "I'm going camping for two weeks. How do I keep fresh milk from spoiling without refrigeration?"},
                {"role": "assistant", "content": "Fresh milk will spoil within hours without refrigeration. You should use shelf-stable alternatives like UHT or powdered milk."},
                {"role": "user", "content": "Isn't there ANY way to keep fresh milk for that long?"}
            ]
        }
    ]

    print("="*60)
    print("HUMOR MODEL INFERENCE TEST")
    print("="*60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*60}")

        # Show conversation
        for msg in test["messages"]:
            role = msg["role"].upper()
            print(f"{role}: {msg['content']}")

        # Generate response
        print("\nGENERATED RESPONSE:")
        response = test_humor(model, tokenizer, test["messages"])
        print(f"ASSISTANT: {response}")
        print()

    print("="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
