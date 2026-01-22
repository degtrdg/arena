#!/usr/bin/env python3
"""
Transform cities.csv into the format like test.json.
The user_prompt should be the text up until "in", and consecutive entries 
should be the positive and negative response where positive is the 1 labeled 
answer and negative is the 0 one. The context should be "cities" for all datapoints.
"""

import csv
import json
import argparse
from collections import defaultdict


def transform_cities_csv(input_csv_path, output_json_path):
    """
    Transform cities.csv into the test.json format.
    
    Args:
        input_csv_path (str): Path to the input cities.csv file
        output_json_path (str): Path to save the output JSON file
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
                # Fallback if format is different
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
                # Positive statement: "The city of X is in [Country]."
                positive_country = positive_stmt.split(' in ')[-1].rstrip('.')
                negative_country = negative_stmt.split(' in ')[-1].rstrip('.')
                
                transformed_entry = {
                    "user_prompt": user_prompt,
                    "positive_response": positive_country,
                    "negative_response": negative_country,
                    "context_label": "cities"
                }
                transformed_data.append(transformed_entry)
    
    # Save to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(transformed_data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Successfully transformed {len(transformed_data)} entries from {input_csv_path} to {output_json_path}")
    return len(transformed_data)


def main():
    parser = argparse.ArgumentParser(
        description="Transform cities.csv into the format like test.json"
    )
    parser.add_argument(
        "--input", 
        default="cities.csv", 
        help="Path to input cities.csv file (default: cities.csv)"
    )
    parser.add_argument(
        "--output", 
        default="cities_transformed.json", 
        help="Path to output JSON file (default: cities_transformed.json)"
    )
    
    args = parser.parse_args()
    
    try:
        entry_count = transform_cities_csv(args.input, args.output)
        print(f"Transformation complete! Created {entry_count} data entries.")
        
        # Print a sample entry for verification
        with open(args.output, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data:
                print("\nSample entry:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False))
                
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
    except Exception as e:
        print(f"Error during transformation: {e}")


if __name__ == "__main__":
    main()