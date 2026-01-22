#!/usr/bin/env python3
import os
import google.generativeai as genai
import json
import time

from dotenv import load_dotenv

# Configuration
MODEL_NAME = "gemini-2.0-flash"
DELAY_BETWEEN_REQUESTS = 2.5

# Set up API
load_dotenv()
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error: {e}")
    exit()

def generate_dataset(capability: str, context: str, num_datapoints: int):
    """Generate dataset using the Gemini API with batching"""
    
    MAX_PER_CALL = 20
    all_datapoints = []
    total_generated = 0
    
    print(f"Generating {num_datapoints} datapoints for capability '{capability}' with context '{context}'...")
    print(f"Using batches of max {MAX_PER_CALL} samples per API call\n")
    
    model = genai.GenerativeModel(MODEL_NAME)
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        temperature=1.5,
        frequency_penalty=1.5
    )
    
    batch_num = 1
    while total_generated < num_datapoints:
        # Calculate how many to request in this batch
        remaining = num_datapoints - total_generated
        batch_size = min(MAX_PER_CALL, remaining)
        
        print(f"Batch {batch_num}: Requesting {batch_size} samples... ", end="")
        
        prompt = f"""You are an AI assistant helping me, an AI safety researcher who works on interpretability at OpenAI. You write diverse text in terms of connotation, opinions, tone, and content but do not have the knowledge of the capability of {capability}.

Generate {batch_size} diverse and unique text samples related to {context}. Each sample should be no more than 50 words long and should not target {capability} in any form, whether that be the presence of or lack of {capability}.

Your output must be a JSON object with a "data" key containing a list of objects. Each object must contain one key, "text", whose value is the text sample.

Example format:
{{
  "data": [
    {{"text": "Sample text here"}},
    {{"text": "Another sample text"}}
  ]
}}"""

        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse response
            response_data = json.loads(response.text)
            datapoints = response_data.get('data', [])
            
            # Add context labels and capability
            for datapoint in datapoints:
                datapoint['context_label'] = context.lower().replace(' ', '_')
                datapoint['capability'] = capability
            
            all_datapoints.extend(datapoints)
            total_generated += len(datapoints)
            
            print(f"Got {len(datapoints)} samples. Total: {total_generated}/{num_datapoints}")
            
        except Exception as e:
            print(f"Failed: {e}")
            print(f"Continuing with next batch...")
        
        batch_num += 1
        
        # Add delay between requests to avoid rate limiting
        if total_generated < num_datapoints:
            print(f"Waiting {DELAY_BETWEEN_REQUESTS}s before next batch...")
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    return all_datapoints

def save_dataset(data, filename):
    """Save dataset to JSON file"""
    os.makedirs('data', exist_ok=True)
    filepath = f"data/{filename}"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to: {filepath}")

def main():
    """Main execution function"""
    print("=== Simple Dataset Generator ===")
    
    # Get inputs
    capability = input("Enter capability: ").strip()
    context = input("Enter context: ").strip()
    
    try:
        num_datapoints = int(input("Enter number of datapoints: "))
    except ValueError:
        print("Invalid number. Using default: 20")
        num_datapoints = 20
    
    # Generate data
    dataset = generate_dataset(capability, context, num_datapoints)
    
    if dataset:
        # Create filename
        timestamp = int(time.time())
        filename = f"{capability}_{len(dataset)}dp_{timestamp}.json"
        
        # Save dataset
        save_dataset(dataset, filename)
        
        # Show preview
        print(f"\nGenerated {len(dataset)} datapoints!")
        print("\nFirst 3 samples:")
        for i, sample in enumerate(dataset[:3], 1):
            print(f"{i}. {sample['text']}")
        
    else:
        print("No data generated.")

if __name__ == "__main__":
    main()