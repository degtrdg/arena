import json

input_file_path = 'data/biology_1000dp.json'
output_file_path = 'data/biology_1000dp_cleaned.json'

allowed_fields = {
    "user_prompt",
    "positive_response",
    "negative_response",
    "context_label"
}

final_cleaned_data = []

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

original_count = len(data)

# A single list comprehension to both filter and clean the data
final_cleaned_data = [
    # 2. For each valid datapoint, create a new dictionary with only the allowed fields
    {key: value for key, value in datapoint.items() if key in allowed_fields}
    # 1. Iterate through the original data and only include datapoints
    #    that have the 'negative_response' field
    for datapoint in data
    if 'negative_response' in datapoint
]

cleaned_count = len(final_cleaned_data)
removed_count = original_count - cleaned_count

with open(output_file_path, 'w', encoding='utf-8') as f:
    # Save the new list of cleaned dictionaries to the output file
    json.dump(final_cleaned_data, f, indent=2)