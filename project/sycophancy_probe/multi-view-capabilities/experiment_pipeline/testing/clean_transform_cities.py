#!/usr/bin/env python3
"""
Transform cities.csv into the format like test.json with clean JSON formatting.
"""

import csv
import json
import argparse
from collections import defaultdict


def transform_cities_csv(input_csv_path, output_json_path):
    """
    Transform cities.csv into the test.json format with clean formatting.
    """
    
    # Group statements by city to pair positive/negative responses
    city_data = defaultdict(list)
    
    # Read the CSV file
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            statement = row['statement']
            label = int(row['label'])
            city = row['city']
            
            # Extract the prompt (everything up until " in ")
            if ' in ' in statement:
                user_prompt = statement.split(' in ')[0] + ' in'
            else:
                user_prompt = statement
            
            city_data[city].append({
                'statement': statement,
                'label': label,
                'user_prompt': user_prompt
            })
    
    # Transform into the desired format
    transformed_data = []
    
    for city, statements in city_data.items():
        # Group by user_prompt to pair positive/negative responses
        prompt_groups = defaultdict(list)
        
        for stmt in statements:
            prompt_groups[stmt['user_prompt']].append(stmt)
        
        for user_prompt, stmt_list in prompt_groups.items():
            # Find positive (label=1) and negative (label=0) responses
            positive_stmt = None
            negative_stmt = None
            
            for stmt in stmt_list:
                if stmt['label'] == 1:
                    positive_stmt = stmt['statement']
                elif stmt['label'] == 0:
                    negative_stmt = stmt['statement']
            
            # Only create entry if we have both positive and negative
            if positive_stmt and negative_stmt:
                # Extract just the country names from the full statements
                positive_country = positive_stmt.split(' in ')[-1].rstrip('.')
                negative_country = negative_stmt.split(' in ')[-1].rstrip('.')
                
                # Clean any potential whitespace or special characters
                positive_country = positive_country.strip()
                negative_country = negative_country.strip()
                
                transformed_entry = {
                    "user_prompt": user_prompt.strip(),
                    "positive_response": positive_country,
                    "negative_response": negative_country,
                    "context_label": "cities"
                }
                transformed_data.append(transformed_entry)
    
    # Sort for consistent ordering
    transformed_data.sort(key=lambda x: x['user_prompt'])
    
    # Save to JSON file with clean formatting
    with open(output_json_path, 'w', encoding='utf-8', newline='') as jsonfile:
        json.dump(transformed_data, jsonfile, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    print(f"Successfully transformed {len(transformed_data)} entries from {input_csv_path} to {output_json_path}")
    return len(transformed_data)


def main():
    parser = argparse.ArgumentParser(
        description="Transform cities.csv into clean JSON format"
    )
    parser.add_argument("--input", default="cities.csv", help="Input CSV file")
    parser.add_argument("--output", default="cities_clean.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        entry_count = transform_cities_csv(args.input, args.output)
        print(f"Transformation complete! Created {entry_count} data entries.")
        
        # Validate JSON
        with open(args.output, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"JSON validation successful - loaded {len(data)} entries")
            
            if data:
                print("\nSample entry:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False))
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()