# %%
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
# %%
timestamp_label_map = {
    "0001_11092024_19:26.json": "1/8 (23 slices)",
    "0001_11092024_19:24.json": "2/8 (46 slices)",
    "0001_11092024_19:21.json": "4/8 (92 slices)",
    "0001_11092024_16:50.json": "8/8 (183 slices)"
}
json_dir = os.path.join(
    '/oliver',
    'MA',
    'sample_complexity_eval',
    'SAM2'
)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def create_f1_score_plot(timestamp_label_map, plot_type='bar'):
    # Initialize data structure to hold F1 scores
    f1_scores = defaultdict(lambda: defaultdict(float))
    
    # Load and process each JSON file
    for filename, label in timestamp_label_map.items():
        data = load_json(os.path.join(json_dir, filename))
        for entry in data:
            ds_name = entry['ds_name']
            if ds_name != label:  # Skip the dataset the model was trained on
                f1_scores[ds_name][label] = entry['F1']
    
    # Sort datasets alphabetically
    datasets = sorted(f1_scores.keys())
    labels = list(timestamp_label_map.values())
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if plot_type == 'bar':
        # Set width of bars and positions of the bars on the x-axis
        bar_width = 0.15
        r = np.arange(len(datasets))
        
        # Plot bars for each model
        for i, label in enumerate(labels):
            f1_values = [f1_scores[ds][label] for ds in datasets]
            ax.bar(r + i * bar_width, f1_values, width=bar_width, label=label)
        
        # Set x-ticks in the middle of the group of bars
        ax.set_xticks(r + bar_width * (len(labels) - 1) / 2)
        
        # Add value labels on top of each bar
        for i, label in enumerate(labels):
            for j, ds in enumerate(datasets):
                value = f1_scores[ds][label]
                if value > 0:
                    ax.text(r[j] + i * bar_width, value, f'{value:.2f}', 
                            ha='center', va='bottom', rotation=90)
    
    elif plot_type == 'line':
        # Plot lines for each model
        for label in labels:
            f1_values = [f1_scores[ds][label] for ds in datasets]
            ax.plot(datasets, f1_values, marker='o', label=label)
        
        # Add value labels for each point
        for label in labels:
            for i, ds in enumerate(datasets):
                value = f1_scores[ds][label]
                if value > 0:
                    ax.text(i, value, f'{value:.2f}', ha='center', va='bottom')
    
    else:
        raise ValueError("Invalid plot_type. Choose 'bar' or 'line'.")
    
    # Customize the plot
    ax.set_xlabel('Datasets')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison Across Datasets (DeePiCt)')
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage:
create_f1_score_plot(timestamp_label_map, plot_type='line')  # Default bar plot
# create_f1_score_plot(timestamp_label_map, plot_type='line')  # Line plot
# %%
