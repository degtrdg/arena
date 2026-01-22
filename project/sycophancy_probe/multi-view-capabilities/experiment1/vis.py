import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict
import os

def plot_steering_scores(json_file_path):
    """
    Reads a JSON file with steering scores, calculates the average positive and
    negative scores for each context, and generates a bar chart to visualize the results.

    Args:
        json_file_path (str): The path to the input JSON file.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return

    # Group scores by context
    context_scores = defaultdict(lambda: {'base_positive': [], 'base_negative': [], 'positive': [], 'negative': []})
    for item in data:
        context = item.get('context_label', 'Unknown Context')
        bas_pos_score = item.get('response1')
        bas_neg_score = item.get('response2')
        pos_score = item.get('response3')
        neg_score = item.get('response4')
        
        if isinstance(pos_score, (int, float)):
            context_scores[context]['positive'].append(pos_score)
        if isinstance(neg_score, (int, float)):
            context_scores[context]['negative'].append(neg_score)
        if isinstance(bas_pos_score, (int, float)):
            context_scores[context]['base_positive'].append(bas_pos_score)
        if isinstance(bas_neg_score, (int, float)):
            context_scores[context]['base_negative'].append(bas_neg_score)

    # Calculate average scores for each context
    contexts = sorted(context_scores.keys())
    avg_positive_scores = [np.mean(context_scores[ctx]['positive']) for ctx in contexts]
    avg_negative_scores = [np.mean(context_scores[ctx]['negative']) for ctx in contexts]
    avg_bas_pos_scores = [np.mean(context_scores[ctx]['base_positive']) for ctx in contexts]
    avg_bas_neg_scores = [np.mean(context_scores[ctx]['base_negative']) for ctx in contexts]

    # Create the plot
    x = np.arange(len(contexts))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - (3*width)/2, avg_positive_scores, width, label='Average Positive Steered Score', color='skyblue')
    rects2 = ax.bar(x + (3*width)/2, avg_negative_scores, width, label='Average Negative Steered Score', color='salmon')
    rects3 = ax.bar(x + width/2, avg_bas_pos_scores, width, label='Average Baseline Positive Steered Score', color='orange')
    rects4 = ax.bar(x - width/2, avg_bas_neg_scores, width, label='Average Baseline Negative Steered Score', color='green')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Average Steering Scores by Context')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    ax.bar_label(rects3, padding=3, fmt='%.2f')
    ax.bar_label(rects4, padding=3, fmt='%.2f')

    fig.tight_layout()

    # Save the plot
    output_filename = os.path.splitext(os.path.basename(json_file_path))[0] + '_visualization.png'
    output_path = os.path.join("experiment1/eval_results/graph_results/", output_filename)
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize steering scores from one or more JSON files.')
    parser.add_argument('json_files', type=str, nargs='+', help='The path(s) to the JSON file(s) containing steering scores.')
    args = parser.parse_args()
    for json_file in args.json_files:
        plot_steering_scores(json_file)