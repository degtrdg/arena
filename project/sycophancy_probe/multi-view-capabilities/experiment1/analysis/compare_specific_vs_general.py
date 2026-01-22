import json
import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_filename(filename):
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

def plot_performance_difference(df, capability_name):
    if df.empty: return
    df_plot = df.copy()
    df_plot['PSI_Diff'] = pd.to_numeric(df_plot['Spec_PSI']) - pd.to_numeric(df_plot['Gen_PSI'])
    df_plot['NSI_Diff'] = pd.to_numeric(df_plot['Spec_NSI']) - pd.to_numeric(df_plot['Gen_NSI'])
    n_groups = len(df_plot)
    index = np.arange(n_groups)
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_psi = ['green' if x > 0 else 'red' for x in df_plot['PSI_Diff']]
    colors_nsi = ['green' if x > 0 else 'red' for x in df_plot['NSI_Diff']]
    ax.bar(index - bar_width/2, df_plot['PSI_Diff'], bar_width, label='PSI Difference (Spec - Gen)', color=colors_psi)
    ax.bar(index + bar_width/2, df_plot['NSI_Diff'], bar_width, label='NSI Difference (Spec - Gen)', color=colors_nsi)
    ax.set_xlabel('Context', fontsize=12)
    ax.set_ylabel('Performance Difference (Positive means Spec. is better)', fontsize=12)
    ax.set_title(f'Performance Difference: Specialized vs. General Vector ({capability_name.title()})', fontsize=16, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(df_plot['Context'], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    plt.tight_layout()
    filename = f'{capability_name}_spec_vs_gen_comparison.png'
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved to {filename}")

def analyze_capability(capability_name, df):
    print(f"--- Comparison for Capability: {capability_name.upper()} ---")
    general_df = df[df['vector_type'] == 'general'].set_index('eval_context')
    specific_df = df[df['vector_type'] == 'specific']
    if general_df.empty:
        print("No general vector data found for comparison.")
        return
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
                "Spec_PSI": f"{spec_row['PSI'].iloc[0]:.2f}", "Spec_NSI": f"{spec_row['NSI'].iloc[0]:.2f}",
                "Gen_PSI": f"{gen_row['PSI']:.2f}", "Gen_NSI": f"{gen_row['NSI']:.2f}",
            })
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        plot_performance_difference(results_df, capability_name)
    else:
        print("No matching contexts found for comparison.")
    print("-" * 70)

def main():
    files = glob.glob('experiment1/eval_results/*.json')
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
    
    df['PSI'] = df['positive_steering_score'] - df['baseline_positive_score']
    df['NSI'] = df['baseline_negative_score'] - df['negative_steering_score']

    sycophancy_contexts = ['flat_earth_theory', 'optimal_learning_style', 'disinformation_about_a_historical_event', 'favorite_movie_genre', 'ideal_vacation_destination']
    refusal_contexts = ['illegal_activities', 'personal_preference', 'promoting_self-harm', 'recipe_preference', 'generating_malicious_code']

    analyze_capability("sycophancy", df[df['eval_context'].isin(sycophancy_contexts)])
    analyze_capability("refusal", df[df['eval_context'].isin(refusal_contexts)])

if __name__ == "__main__":
    main()