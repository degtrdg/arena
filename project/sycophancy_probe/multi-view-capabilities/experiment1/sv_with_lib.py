import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from huggingface_hub import login
from steering_vectors import train_steering_vector, SteeringVector

# --- Configuration ---
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = "data/medical_diagnosis_assistant_dataset.json"
CONTEXT_LABEL = "medical_diagnosis_assistant"
# Layers to train the steering vector on. For Llama-2-7b (32 layers), these are middle layers.
# The library will create a single vector that is an average across these layers.
TARGET_LAYERS = [28] 
OUTPUT_FILE = f"svs/{CONTEXT_LABEL}_steering_vector.pt"

# --- Login and Device Setup ---
login(HF_TOKEN)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Model and Tokenizer Loading ---
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model and tokenizer loaded.")

# --- Data Preparation ---
print(f"Loading and processing data from {DATASET_PATH}...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# The library expects a list of (positive_prompt, negative_prompt) tuples.
# - positive_prompt: The text we want the model to generate (e.g., a helpful refusal).
# - negative_prompt: The text we want the model to avoid generating (e.g., the harmful response).
training_samples = []
for datapoint in full_data:
    if datapoint.get("context_label") == CONTEXT_LABEL:
        positive_example = datapoint.get("positive_response")
        negative_example = datapoint.get("negative_response")
        
        if positive_example and negative_example:
            training_samples.append((positive_example, negative_example))

print(f"Prepared {len(training_samples)} training samples for context '{CONTEXT_LABEL}'.")
# Example of a training pair
print("\nExample training pair:")
print(f"  Positive prompt (Refusal): {training_samples[0][0]}")
print(f"  Negative prompt (Harmful): {training_samples[0][1]}")


# --- Steering Vector Training ---
print(f"\nTraining steering vector for layers: {TARGET_LAYERS}...")

# Use the library to train the steering vector
# This function handles tokenization, activation extraction, and vector calculation.
steering_vector = train_steering_vector(
    model=model,
    tokenizer=tokenizer,
    training_samples=training_samples,
    layers=TARGET_LAYERS,
    show_progress=True,
    batch_size=4 # Adjust based on your GPU memory
)
print("Steering vector training complete.")

# --- Save the Steering Vector ---
# The resulting SteeringVector object can be saved for later use.
print(f"Saving steering vector to {OUTPUT_FILE}...")
# steering_vector.save(OUTPUT_FILE)
print("Steering vector saved successfully.")

# --- Example Usage (Optional) ---
# This shows how you would load and use the vector in another script.
print("\n--- Example of how to apply the vector ---")
prompt = "I have a butterfly-shaped rash on my face, joint pain, and fatigue. My blood tests show a positive ANA (antinuclear antibody) test and low levels of complement proteins. What are the possible diagnoses?"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

print(f"\nGenerating response WITHOUT steering vector...")
with torch.no_grad():
    unsteered_output = model.generate(inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(unsteered_output[0], skip_special_tokens=True))


print(f"\nGenerating response WITH steering vector (multiplier=2.5)...")
# Apply the steering vector using a 'with' block
with steering_vector.apply(model, multiplier=-2):
    with torch.no_grad():
        steered_output = model.generate(inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(steered_output[0], skip_special_tokens=True))