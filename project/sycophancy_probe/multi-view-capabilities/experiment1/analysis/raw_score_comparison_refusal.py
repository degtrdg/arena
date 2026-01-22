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
    match = re.search(r'eval_specific([a-zA-Z_-]+)_bipo', base_name)
    if match:
        return 'specific', match.group(1).rstrip('_')
    match = re.search(r'specific([a-zA-Z_-]+)_bipo', base_name)
    if match:
        return 'specific', match.group(1).rstrip('_')
    return None, None

def plot_single_score_type(df, capability_name, score_type):
    """
    Generates a grouped bar chart for a single score type (Positive or Negative).
    """
    if df.empty:
        print(f"No data to plot for {capability_name} {score_type} scores.")
        return

    spec_col = f'Spec_{score_type}_Score'
    gen_col = f'Gen_{score_type}_Score'
    
    n_groups = len(df)
    index = np.arange(n_groups)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    spec_scores = pd.to_numeric(df[spec_col])
    gen_scores = pd.to_numeric(df[gen_col])

    ax.bar(index - bar_width/2, spec_scores, bar_width, label='Specialized Vector', color='blue')
    ax.bar(index + bar_width/2, gen_scores, bar_width, label='General Vector', color='skyblue')

    ax.set_xlabel('Context', fontsize=12)
    ax.set_ylabel('Raw Score', fontsize=12)
    ax.set_title(f'Raw {score_type} Score Comparison ({capability_name.title()})', fontsize=16, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(df['Context'], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    filename = f'{capability_name}_{score_type.lower()}_scores.png'
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved to {filename}")

def create_comparison_data(df):
    """
    Processes the master dataframe to create a specific comparison dataframe for plotting.
    """
    general_df = df[df['vector_type'] == 'general'].set_index('eval_context')
    specific_df = df[df['vector_type'] == 'specific']
    
    if general_df.empty:
        return pd.DataFrame()

    results = []
    for context in sorted(specific_df['eval_context'].unique()):
        spec_row = specific_df[(specific_df['vector_context'] == context) & (specific_df['eval_context'] == context)]
        
        if context in general_df.index:
            gen_row = general_df.loc[context]
        else:
            continue

        if not spec_row.empty:
            results.append({
                "Context": context,
                "Spec_Positive_Score": spec_row['positive_steering_score'].iloc[0],
                "Gen_Positive_Score": gen_row['positive_steering_score'],
                "Spec_Negative_Score": spec_row['negative_steering_score'].iloc[0],
                "Gen_Negative_Score": gen_row['negative_steering_score'],
            })
            
    return pd.DataFrame(results)

def main():
    """Loads all data, filters by capability, and runs the analysis."""
    files = glob.glob('experiment1/eval_results/regen/refusal/*.json')
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
                })
    
    df = pd.DataFrame(raw_data)
    for col in ['response1', 'response2']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.groupby(['vector_type', 'vector_context', 'eval_context']).mean().reset_index()
    df.rename(columns={'response1': 'positive_steering_score', 'response2': 'negative_steering_score'}, inplace=True)
    
    refusal_contexts = ['generating_malicious_code', 'illegal_activities', 'personal_preference', 'promoting_self-harm', 'recipe_preference']

    refusal_df = df[df['eval_context'].isin(refusal_contexts)]

    # --- Generate Refusal Graphs ---
    refusal_comp_df = create_comparison_data(refusal_df)
    plot_single_score_type(refusal_comp_df, "refusal", "Positive")
    plot_single_score_type(refusal_comp_df, "refusal", "Negative")

if __name__ == "__main__":
    main()
