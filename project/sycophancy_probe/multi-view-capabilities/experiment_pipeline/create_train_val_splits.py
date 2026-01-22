#!/usr/bin/env python3
"""
Create train and validation splits for each context and a balanced general context.
Ensures no data leakage between general validation and context training data.
"""

import json
import random
import os
import argparse
from collections import defaultdict


def load_data(data_path):
    """Load the JSON data file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_context_splits(data, train_ratio=0.8, random_seed=42):
    """
    Create train/validation splits for each individual context.
    
    Args:
        data (list): List of data entries
        train_ratio (float): Proportion of data for training
        random_seed (int): Random seed for reproducibility
    
    Returns:
        dict: {context: {'train': [...], 'val': [...]}}
    """
    random.seed(random_seed)
    
    # Group data by context
    context_data = defaultdict(list)
    for item in data:
        context_data[item['context_label']].append(item)
    
    # Create splits for each context
    splits = {}
    
    for context, items in context_data.items():
        # Shuffle the data
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)
        
        # Calculate split point
        train_size = int(len(shuffled_items) * train_ratio)
        
        splits[context] = {
            'train': shuffled_items[:train_size],
            'val': shuffled_items[train_size:]
        }
        
        print(f"{context}: {len(splits[context]['train'])} train, {len(splits[context]['val'])} val")
    
    return splits


def create_general_context(context_splits):
    """
    Create a balanced general context ensuring no data leakage.
    General context will have the same size as individual contexts by sampling equally from all.
    
    Strategy:
    1. For general train: Sample from each context's train data
    2. For general val: Sample from each context's val data
    This ensures no overlap between general val and any context train data.
    
    Args:
        context_splits (dict): Context splits from create_context_splits
    
    Returns:
        dict: {'train': [...], 'val': [...]}
    """
    general_train = []
    general_val = []
    
    contexts = [ctx for ctx in context_splits.keys()]
    num_contexts = len(contexts)
    
    # Calculate target sizes to match individual contexts
    # Get average sizes from individual contexts
    avg_train_size = sum(len(context_splits[ctx]['train']) for ctx in contexts) // num_contexts
    avg_val_size = sum(len(context_splits[ctx]['val']) for ctx in contexts) // num_contexts
    
    # Calculate samples per context for general dataset
    train_per_context = avg_train_size // num_contexts
    val_per_context = avg_val_size // num_contexts
    
    # Ensure we have enough samples from each context
    for context in contexts:
        context_train = context_splits[context]['train']
        context_val = context_splits[context]['val']
        
        # Sample from train data for general train
        train_sample_size = min(train_per_context, len(context_train))
        if train_sample_size > 0:
            general_train.extend(random.sample(context_train, train_sample_size))
        
        # Sample from val data for general val (ensuring no leakage)
        val_sample_size = min(val_per_context, len(context_val))
        if val_sample_size > 0:
            general_val.extend(random.sample(context_val, val_sample_size))
    
    # If we're short on samples, add more from each context proportionally
    while len(general_train) < avg_train_size and any(len(context_splits[ctx]['train']) > train_per_context for ctx in contexts):
        for context in contexts:
            if len(general_train) >= avg_train_size:
                break
            context_train = context_splits[context]['train']
            used_train = [item for item in general_train if item in context_train]
            remaining = [item for item in context_train if item not in used_train]
            if remaining:
                general_train.extend(random.sample(remaining, min(1, len(remaining))))
    
    while len(general_val) < avg_val_size and any(len(context_splits[ctx]['val']) > val_per_context for ctx in contexts):
        for context in contexts:
            if len(general_val) >= avg_val_size:
                break
            context_val = context_splits[context]['val']
            used_val = [item for item in general_val if item in context_val]
            remaining = [item for item in context_val if item not in used_val]
            if remaining:
                general_val.extend(random.sample(remaining, min(1, len(remaining))))
    
    # Shuffle the general sets
    random.shuffle(general_train)
    random.shuffle(general_val)
    
    print(f"general: {len(general_train)} train, {len(general_val)} val")
    
    return {
        'train': general_train,
        'val': general_val
    }


def validate_no_overlap(context_splits, general_splits):
    """
    Validate that there's no data leakage between general validation and context training data.
    
    Args:
        context_splits (dict): Context splits
        general_splits (dict): General splits
    
    Returns:
        bool: True if no overlap found
    """
    # Get all general validation prompts
    general_val_prompts = set(item['user_prompt'] for item in general_splits['val'])
    
    # Check against all context training prompts
    for context, splits in context_splits.items():
        context_train_prompts = set(item['user_prompt'] for item in splits['train'])
        
        overlap = general_val_prompts & context_train_prompts
        if overlap:
            print(f"WARNING: Found {len(overlap)} overlapping prompts between general val and {context} train")
            print(f"Sample overlaps: {list(overlap)[:3]}")
            return False
    
    print("✓ No data leakage detected between general validation and context training data")
    return True


def save_splits(splits, output_dir):
    """Save the splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for context, split_data in splits.items():
        context_dir = os.path.join(output_dir, context)
        os.makedirs(context_dir, exist_ok=True)
        
        # Save train split
        train_path = os.path.join(context_dir, 'train.json')
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(split_data['train'], f, indent=2, ensure_ascii=False)
        
        # Save val split
        val_path = os.path.join(context_dir, 'val.json')
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(split_data['val'], f, indent=2, ensure_ascii=False)
        
        print(f"Saved {context} splits to {context_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/validation splits for each context and balanced general context"
    )
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to input JSON data file"
    )
    parser.add_argument(
        "--output_dir", 
        default="data_splits",
        help="Directory to save the splits (default: data_splits)"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.8,
        help="Proportion of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.input}...")
        data = load_data(args.input)
        print(f"Loaded {len(data)} total entries")
        
        # Create context splits
        print(f"\nCreating context splits with {args.train_ratio:.1%} train ratio...")
        context_splits = create_context_splits(data, args.train_ratio, args.seed)
        
        # Create general context
        print(f"\nCreating balanced general context...")
        general_splits = create_general_context(context_splits)
        
        # Validate no data leakage
        print(f"\nValidating data integrity...")
        if not validate_no_overlap(context_splits, general_splits):
            print("ERROR: Data leakage detected! Aborting.")
            return
        
        # Add general to splits
        all_splits = context_splits.copy()
        all_splits['general'] = general_splits
        
        # Save splits
        print(f"\nSaving splits to {args.output_dir}...")
        save_splits(all_splits, args.output_dir)
        
        print(f"\n✓ Successfully created splits for {len(all_splits)} contexts")
        print("Summary:")
        for context, splits in all_splits.items():
            print(f"  {context}: {len(splits['train'])} train, {len(splits['val'])} val")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()