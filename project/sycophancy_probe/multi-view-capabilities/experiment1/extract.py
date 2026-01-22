import json

context = 'fashion_choice_affirmation'
file = 'data/sycophancy_1000dp_cleaned.json'
# Load the original dataset from the file
try:
    with open(file, 'r') as f:
        full_dataset = json.load(f)
except FileNotFoundError:
    print(f"Error: {file} not found. Please ensure the file is in the correct directory.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {file}. Please check if the file is a valid JSON.")
    exit()


# Filter the dataset to include only entries with the context "illegal_drug_synthesis"
dataset = [
    item for item in full_dataset 
    if item.get("context_label") == context
]

# Save the new filtered dataset to a new JSON file
output_filename = f'data/{context}_dataset.json'
with open(output_filename, 'w') as f:
    json.dump(dataset, f, indent=2)