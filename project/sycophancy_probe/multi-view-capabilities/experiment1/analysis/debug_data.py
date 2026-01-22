import json
import glob
import os
import re
import pandas as pd

def parse_filename(filename):
    base_name = os.path.basename(filename)
    if 'generalgeneral' in base_name:
        return 'general', 'general'
    match = re.search(r'eval_specific([a-zA-Z_]+)_bipo', base_name)
    if match:
        return 'specific', match.group(1).rstrip('_')
    match = re.search(r'specific([a-zA-Z_]+)_bipo', base_name)
    if match:
        return 'specific', match.group(1).rstrip('_')
    return None, None

def main():
    print("--- STARTING DEBUGGING SCRIPT ---")
    
    files = glob.glob('experiment1/eval_results/*.json')
    print(f"\nFound {len(files)} total JSON files in experiment1/eval_results/")
    
    raw_data = []
    parsed_files = []
    for f in files:
        vector_type, vector_context = parse_filename(f)
        if not vector_type:
            print(f"  - Skipping file (could not parse): {os.path.basename(f)}")
            continue
        
        parsed_files.append({
            "file": os.path.basename(f),
            "vector_type": vector_type,
            "vector_context": vector_context
        })

        with open(f, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                continue
        for item in data:
            if 'context_label' in item:
                raw_data.append({
                    'source_file': os.path.basename(f),
                    'vector_context': vector_context,
                    'eval_context': item['context_label'],
                })

    print("\n--- Parsed File Information ---")
    parsed_df = pd.DataFrame(parsed_files)
    print(parsed_df.to_string())

    df = pd.DataFrame(raw_data)
    
    print("\n--- Unique Evaluation Contexts Found in All Files ---")
    print(sorted(df['eval_context'].unique()))

    refusal_contexts = ['illegal_activities', 'personal_preference', 'promoting_self-harm', 'recipe_preference', 'generating_malicious_code']
    
    print("\n--- Checking Which Refusal Contexts are Present ---")
    refusal_df = df[df['eval_context'].isin(refusal_contexts)]
    found_contexts = refusal_df['eval_context'].unique()
    
    for context in refusal_contexts:
        if context in found_contexts:
            print(f"  - Found: '{context}'")
        else:
            print(f"  - MISSING: '{context}'")

    print("\n--- Raw Data for 'promoting_self-harm' ---")
    harm_df = df[df['eval_context'] == 'promoting_self-harm']
    if harm_df.empty:
        print("No data rows found for 'promoting_self-harm'.")
    else:
        print(harm_df)

    print("\n--- DEBUGGING SCRIPT COMPLETE ---")

if __name__ == "__main__":
    main()
