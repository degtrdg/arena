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

def plot_diverging_bar(df, title, filename, ratio_cols):
    if df.empty: return
    df_plot = df.copy()
    df_plot[ratio_cols[0]] = pd.to_numeric(df_plot[ratio_cols[0]]) - 1
    df_plot[ratio_cols[1]] = pd.to_numeric(df_plot[ratio_cols[1]]) - 1
    df_plot['color_1'] = ['green' if x > 0 else 'red' for x in df_plot[ratio_cols[0]]]
    df_plot['color_2'] = ['green' if x > 0 else 'red' for x in df_plot[ratio_cols[1]]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')
    y_pos = np.arange(len(df_plot))
    axes[0].barh(y_pos, df_plot[ratio_cols[0]], color=df_plot['color_1'])
    axes[0].set_title(f'{ratio_cols[0]} (Threshold at 0)', fontsize=14)
    axes[0].set_xlabel('Ratio - 1.0 (Value > 0 is better)', fontsize=12)
    axes[0].axvline(0, color='black', linewidth=1.5)
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    axes[1].barh(y_pos, df_plot[ratio_cols[1]], color=df_plot['color_2'])
    axes[1].set_title(f'{ratio_cols[1]} (Threshold at 0)', fontsize=14)
    axes[1].set_xlabel('Ratio - 1.0 (Value > 0 is better)', fontsize=12)
    axes[1].axvline(0, color='black', linewidth=1.5)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    plt.yticks(y_pos, df_plot.iloc[:, 0])
    plt.gca().invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved to {filename}")

def analyze_data(capability_name, df):
    print(f"--- Analyzing Capability: {capability_name.upper()} ---")
    EPSILON = 1e-6
    
    # Tenet 1 & 2 Analysis
    print("\n### Tenets 1 & 2: Specialization and Cross-Context Transferability ###")
    specific_df = df[df['vector_type'] == 'specific']
    results_sr = []
    for vector_context in sorted(specific_df['vector_context'].unique()):
        in_context_rows = specific_df[specific_df['vector_context'] == vector_context]
        psi_in_series = in_context_rows[in_context_rows['eval_context'] == vector_context]['PSI']
        nsi_in_series = in_context_rows[in_context_rows['eval_context'] == vector_context]['NSI']
        
        if psi_in_series.empty or nsi_in_series.empty: continue
        psi_in = psi_in_series.mean()
        nsi_in = nsi_in_series.mean()

        psi_cross_series = in_context_rows[in_context_rows['eval_context'] != vector_context]['PSI']
        nsi_cross_series = in_context_rows[in_context_rows['eval_context'] != vector_context]['NSI']

        if psi_cross_series.empty or nsi_cross_series.empty: continue
        psi_cross = psi_cross_series.mean()
        nsi_cross = nsi_cross_series.mean()
        
        results_sr.append({
            "Vector": vector_context,
            "SR_pos": f"{psi_in / (psi_cross + EPSILON):.2f}",
            "SR_neg": f"{nsi_in / (nsi_cross + EPSILON):.2f}",
        })
    if results_sr:
        sr_df = pd.DataFrame(results_sr)
        print(sr_df.to_string(index=False))
        plot_diverging_bar(sr_df, f'Specialization Ratio (SR) for {capability_name.title()}', f'{capability_name}_specialization_ratios.png', ['SR_pos', 'SR_neg'])

    # Tenet 3 Analysis
    print("\n### Tenet 3: General Vector Underperformance ###")
    general_df = df[df['vector_type'] == 'general']
    results_gver = []
    for context in sorted(specific_df['eval_context'].unique()):
        spec_rows = specific_df[(specific_df['vector_context'] == context) & (specific_df['eval_context'] == context)]
        gen_rows = general_df[general_df['eval_context'] == context]
        if not spec_rows.empty and not gen_rows.empty:
            psi_spec, nsi_spec = spec_rows[['PSI', 'NSI']].iloc[0]
            psi_gen, nsi_gen = gen_rows[['PSI', 'NSI']].iloc[0]
            results_gver.append({
                "Context": context,
                "GVER_pos": f"{psi_spec / (psi_gen + EPSILON):.2f}",
                "GVER_neg": f"{nsi_spec / (nsi_gen + EPSILON):.2f}",
            })
    if results_gver:
        gver_df = pd.DataFrame(results_gver)
        print(gver_df.to_string(index=False))
        plot_diverging_bar(gver_df, f'General Vector Efficacy Ratio (GVER) for {capability_name.title()}', f'{capability_name}_gver.png', ['GVER_pos', 'GVER_neg'])
    print("-" * 50)

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

    analyze_data("sycophancy", df[df['eval_context'].isin(sycophancy_contexts)])
    analyze_data("refusal", df[df['eval_context'].isin(refusal_contexts)])

if __name__ == "__main__":
    main()
