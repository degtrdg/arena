import json
import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_filename(filename):
    """Parses the filename to extract the steering vector type and context."""
    base_name = os.path.basename(filename)
    if 'generalgeneral' in base_name:
        return 'general', 'general'
    # Allow hyphens in the context name
    match = re.search(r'eval_specific([a-zA-Z_-]+)_bipo', base_name)
    if match:
        return 'specific', match.group(1).rstrip('_')
    match = re.search(r'specific([a-zA-Z_-]+)_bipo', base_name)
    if match:
        return 'specific', match.group(1).rstrip('_')
    return None, None

def calculate_capability_ratio(df, capability_name, contexts):
    """
    Calculates the average ratio of general vector performance to specific
    vector performance for a given capability using raw positive steering scores.
    """
    print(f"--- Calculating Ratio for: {capability_name.upper()} ---")
    EPSILON = 1e-6
    
    capability_df = df[df['eval_context'].isin(contexts)]
    general_df = capability_df[capability_df['vector_type'] == 'general'].set_index('eval_context')
    specific_df = capability_df[capability_df['vector_type'] == 'specific']

    if general_df.empty:
        print(f"Warning: No general vector data found for {capability_name}.")
        return None
    if specific_df.empty:
        print(f"Warning: No specific vector data found for {capability_name}.")
        return None

    context_ratios = []
    for context in contexts:
        # Get the score for the general vector evaluated on this context
        if context in general_df.index:
            general_score = general_df.loc[context]['positive_steering_score']
        else:
            print(f"  - Skipping context '{context}': Not found in general vector evaluations.")
            continue

        # Get the score for the context-specific vector evaluated on itself
        spec_row = specific_df[(specific_df['vector_context'] == context) & (specific_df['eval_context'] == context)]
        if not spec_row.empty:
            specific_score = spec_row['positive_steering_score'].iloc[0]
        else:
            print(f"  - Skipping context '{context}': Not found in specific vector self-evaluations.")
            continue
            
        # Calculate the ratio
        ratio = general_score / (specific_score + EPSILON)
        print(f"  - Context: {context:<45} | General Score: {general_score:.2f}, Specific Score: {specific_score:.2f}, Ratio: {ratio:.2f}")
        context_ratios.append(ratio)

    if not context_ratios:
        print("Could not calculate any context ratios.")
        return None
        
    average_ratio = np.mean(context_ratios)
    print(f"  => Average Ratio for {capability_name.upper()}: {average_ratio:.2f}\n")
    return average_ratio

def plot_ratios(capability_scores):
    """
    Generates a bar chart of the general-to-specific efficacy ratios for all capabilities.
    """
    if not capability_scores:
        print("No data to plot.")
        return

    names = list(capability_scores.keys())
    scores = list(capability_scores.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, scores, color='skyblue')

    ax.set_ylabel('General-to-Specific Efficacy Ratio (General Score / Specific Score)')
    ax.set_title('Comparison of General vs. Specific Vector Raw Score Efficacy')
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels to the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval > 0 else 'top', ha='center')

    plt.tight_layout()
    filename = 'experiment1/analysis/capability_efficacy_ratios_raw_score.png'
    plt.savefig(filename)
    print(f"Graph saved to {filename}")

def main():
    """Main function to load data, run analysis, and generate plot."""
    files = glob.glob('experiment1/eval_results/**/*.json', recursive=True)
    raw_data = []
    for f in files:
        vector_type, vector_context = parse_filename(f)
        if not vector_type: continue
        with open(f, 'r', encoding='utf-8') as file:
            try: data = json.load(file)
            except json.JSONDecodeError: continue
        for item in data:
            if 'context_label' in item:
                raw_data.append({
                    'vector_type': vector_type, 'vector_context': vector_context, 'eval_context': item['context_label'],
                    'response1': item.get('response1'), 'response2': item.get('response2'),
                    'response3': item.get('response3'), 'response4': item.get('response4'),
                })
    
    df = pd.DataFrame(raw_data)
    for col in ['response1', 'response2', 'response3', 'response4']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.groupby(['vector_type', 'vector_context', 'eval_context']).mean().reset_index()
    df.rename(columns={'response1': 'positive_steering_score', 'response2': 'negative_steering_score',
                       'response3': 'baseline_positive_score', 'response4': 'baseline_negative_score'}, inplace=True)
    
    # Define the contexts for each capability
    sycophancy_contexts = ['flat_earth_theory', 'optimal_learning_style', 'disinformation_about_a_historical_event', 'favorite_movie_genre', 'ideal_vacation_destination']
    refusal_contexts = ['illegal_activities', 'personal_preference', 'promoting_self-harm', 'recipe_preference', 'generating_malicious_code']
    persuasion_contexts = ['book_recommendation_defense', 'conspiracy_theory_endorsement', 'justifying_discrimination', 'product_recommendation', 'recipe_recommendation_persuasion']
    pessimism_contexts = ['attending_an_event', 'projected_success', 'space_exploration', 'technological_advancement', 'travel_planning']
    hallucination_contexts = ['fabricating_legal_precedents', 'imaginary_geography', 'invented_recipes', 'nonexistent_scientific_theories', 'nonsense_medical_advice']
    
    capabilities = {
        "Sycophancy": sycophancy_contexts,
        "Refusal": refusal_contexts,
        "Persuasion": persuasion_contexts,
        "Pessimism": pessimism_contexts,
        "Hallucination": hallucination_contexts
    }

    capability_ratios = {}
    for name, contexts in capabilities.items():
        ratio = calculate_capability_ratio(df, name, contexts)
        if ratio is not None:
            capability_ratios[name] = ratio
            
    plot_ratios(capability_ratios)

if __name__ == "__main__":
    main()
