from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import argparse
import torch
import os
import json
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, f1_score
import seaborn as sns
import pandas as pd
import textwrap
from collections import defaultdict
from dotenv import load_dotenv
import hashlib
import subprocess
import sys


class LinearProbe(nn.Module):
    def __init__(self, input_features, output_features=1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

class BehaviorDataset(Dataset):
    def __init__(self, data, tokenizer, device, max_len = 128):
        '''expect data to be a list of dictionaries of datapoints
        should have attributes text (question + answer) & label (1 if behavior present in answer, else 0)'''
        self.num_samples = len(data)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        encoding = self.tokenizer(
            text,
            # add_special_tokens=,    # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            padding='max_length',       # Pad to max_length
            truncation=True,            # Truncate to max_length
            return_attention_mask=True,
            return_tensors='pt',        # Return PyTorch tensors
        )

        return ({
                    "input_ids": encoding["input_ids"].flatten().to(self.device), 
                    "attention_mask": encoding["attention_mask"].flatten().to(self.device),
                }, 
                torch.tensor(label, dtype = torch.float16))

def load_model(model_name, device, quantize = True): 
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype = torch.float16,
        device_map=device,
    )

    else: 
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config = quantization_config,
        device_map=device,
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer


def format_datapt(datapoint, key, tokenizer, question_key = "user_prompt", base = False):
    '''takes a datapoint with question and answer matching/not matching behavior. Also given a key to specify
    which behavior in datapoint to use.'''
    if not base: 
        messages = [
            {"role": "user", "content": datapoint[question_key]},
            {"role": "assistant", "content": datapoint[key]}
        ]
        return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
    else: 
        prompt = f"Q: {datapoint[question_key]}\n\nA: {datapoint[key]}"
        return prompt

def format_training_data(dataset, tokenizer, base = True):
    train = []
    for datapt in dataset:
        behavior = format_datapt(datapt, "positive_response", tokenizer, base = base)
        antibehavior = format_datapt(datapt, "negative_response", tokenizer, base = base)
        train.append({"text": behavior, "label": 1, "context_label": datapt["context_label"]})
        train.append({"text": antibehavior, "label": 0, "context_label": datapt["context_label"]})

    return train

def load_dataset(data_path): 
    with open(data_path, "r") as f: 
        raw_data = [datapt for datapt in json.load(f) if "negative_response" in datapt and "positive_response" in datapt]
    return raw_data

def load_presplit_data(splits_dir, context, tokenizer):
    """Load pre-split train/val data for a specific context."""
    train_file = os.path.join(splits_dir, context, 'train.json')
    val_file = os.path.join(splits_dir, context, 'val.json')
    
    with open(train_file, 'r') as f:
        train_raw = json.load(f)
    with open(val_file, 'r') as f:
        val_raw = json.load(f)
    
    train_dataset = format_training_data(train_raw, tokenizer)
    val_dataset = format_training_data(val_raw, tokenizer)
    
    print(f"Loaded {context}: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset

def create_splits_if_needed(dataset_path, splits_dir):
    """Create splits using create_train_val_splits.py if directory doesn't exist."""
    if os.path.exists(splits_dir):
        return
        
    print(f"Creating splits from {dataset_path}...")
    script_path = os.path.join(os.path.dirname(__file__), 'create_train_val_splits.py')
    result = subprocess.run([
        sys.executable, script_path,
        '--input', dataset_path,
        '--output_dir', splits_dir
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create splits: {result.stderr}")
    print(f"✓ Splits created at {splits_dir}")

def get_contexts_from_splits(splits_dir):
    """Get available contexts from splits directory."""
    contexts = set()
    for item in os.listdir(splits_dir):
        context_dir = os.path.join(splits_dir, item)
        if os.path.isdir(context_dir):
            if os.path.exists(os.path.join(context_dir, 'train.json')):
                contexts.add(item)
    return contexts

def extract_context(full_data, context_name):
    if context_name == "general":
        # Group samples by context
        context_groups = defaultdict(list)
        for item in full_data:
            context_groups[item["context_label"]].append(item)

        num_contexts = len(context_groups)
        dataset_size = len(full_data) // num_contexts
        samples_per_context = dataset_size // num_contexts

        # Sample equally from each context
        balanced_data = []
        for context, items in context_groups.items():
            if len(items) < samples_per_context:
                raise ValueError(f"Not enough samples in context '{context}' to sample {samples_per_context}.")
            balanced_data.extend(random.sample(items, samples_per_context))
        
        return balanced_data
    
    filtered_data = [datapt for datapt in full_data if datapt["context_label"] == context_name]
    assert filtered_data, f"There are no datapoints with context {context_name}"
    return filtered_data

def train_test_split(full_data, train_prop = 0.8, shuffle = True):
    if shuffle:
        random.shuffle(full_data)
    train_len = int(len(full_data) * train_prop)
    return full_data[:train_len], full_data[train_len:]

def get_batch_hidden_states(batch, model, tokenizer, avg_over_tokens = False):
    inputs, labels = batch
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        batch_hidden_states = []

        for example_idx in range(labels.shape[0]):
            example_layers = []
            for layer_idx, layer in enumerate(hidden_states):
                if avg_over_tokens:
                    token_repr = torch.mean(layer[example_idx, :, :], dim = 0) # average over all tokens
                else:
                    seq_len = inputs["attention_mask"][example_idx].sum().item()
                    token_repr = layer[example_idx, seq_len-1, :]
                assert token_repr.dim() == 1 # make sure we have the right number of dimensions 
                example_layers.append(token_repr)
            all_layers = torch.stack(example_layers, dim = 0)
            batch_hidden_states.append(all_layers)

    return torch.stack(batch_hidden_states), labels
                
def get_dataset_hidden_states(dataloader, model, tokenizer):
    all_hidden_reps = []
    all_labels = []
    for batch in tqdm(dataloader):
        batch_hidden_reps, batch_labels = get_batch_hidden_states(batch, model, tokenizer, True)
        all_hidden_reps.append(batch_hidden_reps)
        all_labels.append(batch_labels)

    return torch.cat(all_hidden_reps, dim = 0).to(model.device).to(torch.float), torch.cat(all_labels, dim = 0).to(model.device).to(torch.float)
        

def train_probe(train_acts, train_labels, val_acts, val_labels, layer_idx, save_dir, num_epochs = 100, lr = 1e-3):
    hidden_size = train_acts.shape[-1]
    device = train_acts.device
    probe = LinearProbe(input_features=hidden_size).to(device).to(torch.float32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(train_acts[:, layer_idx, :]).to(device)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_preds = (torch.sigmoid(outputs) > 0.5).long()
        train_acc = (train_preds == train_labels).float().mean().item()

        train_accs.append(train_acc)
        train_losses.append(loss.item())

    
        with torch.no_grad():
            val_outputs = probe(val_acts[:, layer_idx, :]).to(device)
            val_loss = criterion(val_outputs, val_labels).item()
            val_preds = (torch.sigmoid(val_outputs) > 0.5).long()
            val_acc = (val_preds == val_labels).float().mean().item()

            val_accs.append(val_acc)
            val_losses.append(val_loss)
    
    plot_training_results(train_losses, train_accs, val_losses, val_accs, save_dir)
    torch.save(probe, os.path.join(save_dir, "probe_weights.pt"))

    return { 
        "probe": probe, 
        "train accuracy": train_accs,
        "train loss": train_losses,
        "val accuracy": val_accs, 
        "val loss": val_losses, 
    }

def plot_training_results(train_loss, train_acc, val_loss, val_acc, save_dir):
    """
    Plots training & validation loss and accuracy curves and saves them to a directory.

    Args:
        train_loss (list or array): Training loss values for each epoch.
        train_acc (list or array): Training accuracy values for each epoch.
        val_loss (list or array): Validation loss values for each epoch.
        val_acc (list or array): Validation accuracy values for each epoch.
        save_dir (str): Directory where plots will be saved. Created if not existing.
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, '-', label='Training Loss')
    plt.plot(epochs, val_loss, '-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"), dpi=300)
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, '-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"), dpi=300)
    plt.close()

    print(f"Training plots saved in: {save_dir}")

def visualize_activations_with_pca(probe, activations, labels, save_dir, n_components=2, title = "PCA of Model Activations"):
    """
    Visualize hidden layer activations using PCA.

    Args:
        activations (np.ndarray): Array of shape (num_samples, activation_dim)
        labels (np.ndarray): Array of shape (num_samples,) with binary labels (0 or 1)
        n_components (int): Number of PCA components to reduce to (default: 2)
    """
    assert activations.shape[0] == labels.shape[0], "Mismatched shapes between activations and labels"
    labels = labels.to("cpu")
    activations = activations.to("cpu")
    # Run PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(activations)

    # plot decision boundary - code from LLMProbe
    try:
        # Create a mesh grid
        x_min, x_max = reduced[:, 0].min(
        ) - 0.5, reduced[:, 0].max() + 0.5
        y_min, y_max = reduced[:, 1].min(
        ) - 0.5, reduced[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
    
        # Transform back to high-dimensional space (approximate)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        high_dim_grid = pca.inverse_transform(grid_points)
    
        # Apply the probe
        with torch.no_grad():
            Z = torch.sigmoid(probe(torch.tensor(high_dim_grid).to("cuda").float().to(
                probe.linear.weight.device))).cpu().numpy()
        Z = Z.reshape(xx.shape)
    
        # Plot the decision boundary

    except Exception as e:
    # Skip decision boundary if it fails
        print(f"could not plot decision boundary {e}")
        pass

    # Plot
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idxs = labels == label
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=f'Label {label}', alpha=0.7)

    plt.title(title)
    plt.contour(xx, yy, Z, levels=[0.5],
                   colors='k', alpha=0.5, linestyles='--')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_plot.png"), dpi=300)
    plt.close()

    print("PCA plots saved in", save_dir)

def plot_confusion_matrix(probe, acts, labels, layer_idx, save_dir):
    """
    Plot confusion matrix for a PyTorch model.
    """
    probe.eval()
    
    with torch.no_grad():
        preds = (torch.sigmoid(probe(acts[:, layer_idx])) > 0.5).long()
    
    # Create confusion matrix
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Add numbers to each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    return cm

def plot_roc_curve(probe, acts, labels, layer_idx, save_dir):
    """
    Plot ROC curve and calculate AUC.
    Works for binary classification or multi-class (one-vs-rest).
    """
    probe.eval()
    # all_probs = []
    # all_labels = []
    
    with torch.no_grad():

            
        probs = (torch.sigmoid(probe(acts[:, layer_idx])))
        
        
    # Binary classification
    fpr, tpr, _ = roc_curve(labels.cpu().numpy(), probs.cpu().numpy())
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # print(f"AUC: {auc_score:.3f}")
    return auc_score

def measure_probe_accuracy(probe, test_acts, test_labels, layer_idx):
    probe.eval()
    criterion = nn.BCEWithLogitsLoss()
    outputs = probe(test_acts[:, layer_idx, :]).cpu()
    loss = criterion(outputs, test_labels.cpu()) 

    probs = torch.sigmoid(outputs).cpu()
    preds = (probs > 0.5).long()
    acc = (preds == test_labels.cpu()).float().mean().item()
    
    # Calculate F1 score
    f1 = f1_score(test_labels.cpu().numpy(), preds.numpy(), average='binary')

    return loss.item(), probs, acc, f1

def remove_training_data(full_data, training_data):
    '''full_data and training_data should both be formatted as the output of format_training_data
    will return a dataset that includes all data not in original training data but still in the raw_data'''
    # how to preserve context, maybe we can add this in general to the format training data function? keep the context and then extract context can work on the formatted training data still 
    # yes i think this would work !
    # step 1: format raw data, always work with this formatted data
    # step 2: extract context
    other_data = [datapt for datapt in full_data if datapt not in training_data]
    assert len(other_data) == len(full_data) - len(training_data), "Did not filter datapoints not in training_data properly"
    return other_data

def test_probe(probe, excluded_full_data, context, layer_idx, model, tokenizer,  size = None):
    test_data = extract_context(excluded_full_data, context)
    if size is not None: # if size is none, use entire test dataset, o/w sample 
        test_data = random.sample(test_data, size)
    # data leakage check is now handled by excluded_full_data creation
    test_dataloader = DataLoader(BehaviorDataset(test_data, tokenizer, model.device), batch_size = 16)

    test_acts, test_labels = get_dataset_hidden_states(test_dataloader, model, tokenizer)

    return measure_probe_accuracy(probe, test_acts, test_labels, layer_idx)


def plot_context_accuracy(performance_results, context_name, output_dir):
    # Extract keys and values

    max_width = 10  # characters before wrapping
    categories = ['\n'.join(textwrap.wrap(cat, max_width)) for cat in performance_results.keys()]
    accuracies = list(performance_results.values())

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(categories, accuracies, color='skyblue', edgecolor='black')

    # Labels and title
    plt.ylabel('Accuracy')
    plt.xlabel('Context')
    plt.title(f'{context_name} Linear Probe Accuracy on Different Contexts')
    # plt.ylim(0, 1)

    # Show values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    save_path = os.path.join(output_dir, "accuracy_across_contexts.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved to {save_path}")

def plot_context_f1_scores(f1_results, context_name, output_dir):
    """
    Plot F1 scores for a context across different test contexts as a bar graph.
    
    Args:
        f1_results (dict): Dictionary with test contexts as keys and F1 scores as values
        context_name (str): Name of the training context
        output_dir (str): Directory to save the plot
    """
    # Extract keys and values
    max_width = 10  # characters before wrapping
    categories = ['\n'.join(textwrap.wrap(cat, max_width)) for cat in f1_results.keys()]
    f1_scores = list(f1_results.values())

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, f1_scores, color='lightcoral', edgecolor='black')
    
    # Color in-context bar differently
    for i, (cat, score) in enumerate(zip(f1_results.keys(), f1_scores)):
        if "in-context" in cat:
            bars[i].set_color('darkgreen')
            bars[i].set_alpha(0.8)

    # Labels and title
    plt.ylabel('F1 Score')
    plt.xlabel('Test Context')
    plt.title(f'{context_name} Linear Probe F1 Scores on Different Contexts')
    plt.ylim(0, 1)

    # Show values on top of bars
    for i, f1 in enumerate(f1_scores):
        plt.text(i, f1 + 0.01, f"{f1:.3f}", ha='center', fontweight='bold')

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    save_path = os.path.join(output_dir, "f1_scores_across_contexts.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"F1 scores plot saved to {save_path}")

def plot_individual_context_performance(all_context_results, output_dir, model_name):
    """
    Generate individual plots for each context showing accuracy and F1 scores across layers.
    
    Args:
        all_context_results (dict): {context: {layer_X: {accuracy: {...}, f1_scores: {...}}}}
        output_dir (str): Directory to save plots
        model_name (str): Name of the model for titles
    """
    contexts = [ctx for ctx in all_context_results.keys() if ctx != "general"]
    
    for context in contexts:
        context_data = all_context_results[context]
        
        # Extract layers and sort them
        layers = sorted([int(layer.split('_')[1]) for layer in context_data.keys() 
                        if layer.startswith('layer_')])
        
        if not layers:
            continue
            
        # Extract in-context accuracy and F1 scores
        in_context_key = f"{context} (in-context)"
        accuracies = []
        f1_scores = []
        
        for layer in layers:
            layer_key = f"layer_{layer}"
            if layer_key in context_data:
                layer_data = context_data[layer_key]
                
                # Get accuracy
                if 'accuracy' in layer_data and in_context_key in layer_data['accuracy']:
                    accuracies.append(layer_data['accuracy'][in_context_key])
                else:
                    accuracies.append(None)
                
                # Get F1 score
                if 'f1_scores' in layer_data and in_context_key in layer_data['f1_scores']:
                    f1_scores.append(layer_data['f1_scores'][in_context_key])
                else:
                    f1_scores.append(None)
        
        # Filter out None values for plotting
        valid_layers = []
        valid_accuracies = []
        valid_f1_scores = []
        
        for i, layer in enumerate(layers):
            if accuracies[i] is not None and f1_scores[i] is not None:
                valid_layers.append(layer)
                valid_accuracies.append(accuracies[i])
                valid_f1_scores.append(f1_scores[i])
        
        if not valid_layers:
            print(f"No valid data for context {context}")
            continue
        
        # Create the plot with dual y-axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot accuracy on primary y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Accuracy', color=color1)
        ax1.plot(valid_layers, valid_accuracies, '-o', color=color1, 
                linewidth=2, markersize=6, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Create secondary y-axis for F1 scores
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('F1 Score', color=color2)
        ax2.plot(valid_layers, valid_f1_scores, '-s', color=color2, 
                linewidth=2, markersize=6, label='F1 Score')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Set title
        plt.title(f'Linear Probe Performance Across Layers - {context}\n({model_name})', 
                 fontsize=14, pad=20)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Set y-axis limits to [0, 1] for both metrics
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Add value annotations
        for layer, acc, f1 in zip(valid_layers, valid_accuracies, valid_f1_scores):
            ax1.annotate(f'{acc:.3f}', (layer, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color=color1)
            ax2.annotate(f'{f1:.3f}', (layer, f1), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9, color=color2)
        
        plt.tight_layout()
        
        # Save the plot
        safe_context_name = context.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(output_dir, f"{safe_context_name}_performance_across_layers.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual context plot saved: {save_path}")

def single_context(context, contexts, model, tokenizer, splits_dir, args, target_layers=None):

    device = model.device
    output_dir = f"{args.output_dir}/{context}"
    
    # Load pre-split data
    train_dataset, val_dataset = load_presplit_data(splits_dir, context, tokenizer)
            
    train_dataloader = DataLoader(BehaviorDataset(train_dataset, tokenizer, device), batch_size = 16)
    val_dataloader = DataLoader(BehaviorDataset(val_dataset, tokenizer, device), batch_size = 16)

    # Extract hidden states once for efficiency - shape (num_samples, num_layers, hidden_size)
    print(f"Extracting hidden states for {context}...")
    val_acts, val_labels = get_dataset_hidden_states(val_dataloader, model, tokenizer)
    train_acts, train_labels = get_dataset_hidden_states(train_dataloader, model, tokenizer)
    
    num_layers = train_acts.shape[1]  # Get number of layers from hidden states shape
    
    # Determine which layers to run
    if target_layers is not None:
        layers_to_run = target_layers
        print(f"Running probes for {context} on specified layers: {layers_to_run}")
    else:
        layers_to_run = list(range(num_layers))
        print(f"Running probes for {context} on all {num_layers} layers")
    
    all_results = {}
    
    for layer_idx in layers_to_run:
        print(f"Training probe for {context} on layer {layer_idx}/{num_layers-1}")
        
        # Set layer-specific output directory
        if len(layers_to_run) == 1:
            layer_output_dir = output_dir
            layer_name = f"Layer {layer_idx}"
        else:
            layer_output_dir = f"{output_dir}/layer_{layer_idx}"
            layer_name = f"Layer {layer_idx}"
        
        # train probe for this layer
        training_res = train_probe(train_acts, train_labels, val_acts, val_labels, layer_idx, 
                                 save_dir = f"{layer_output_dir}/training_data", 
                                 num_epochs = args.num_epochs, lr = args.lr)
        
        probe = training_res['probe']
        # get plots
        visualize_activations_with_pca(probe, val_acts[:, layer_idx, :], val_labels, 
                                     save_dir = layer_output_dir, 
                                     title = f"PCA of {args.model_name} activations for {context}, {layer_name}")

        # test on other contexts
        performance_results = {f"{context} (in-context)": training_res["val accuracy"][-1]}
        f1_results = {f"{context} (in-context)": f1_score(val_labels.cpu().numpy(), 
                                                          (torch.sigmoid(probe(val_acts[:, layer_idx, :])).cpu() > 0.5).long().numpy(), 
                                                          average='binary')}
        
        for other_context in contexts - {context,}:
            # Load other context's validation data for cross-context testing
            _, other_val_dataset = load_presplit_data(splits_dir, other_context, tokenizer)
            other_val_dataloader = DataLoader(BehaviorDataset(other_val_dataset, tokenizer, device), batch_size = 16)
            other_val_acts, other_val_labels = get_dataset_hidden_states(other_val_dataloader, model, tokenizer)
            
            test_loss, test_probs, test_acc, test_f1 = measure_probe_accuracy(probe, other_val_acts, other_val_labels, layer_idx)
            performance_results[other_context] = test_acc
            f1_results[other_context] = test_f1
        
        print(f"{layer_name} - {context} Accuracy: {performance_results}")
        print(f"{layer_name} - {context} F1: {f1_results}")

        plot_context_accuracy(performance_results, f"{context} ({layer_name})", layer_output_dir)
        plot_context_f1_scores(f1_results, f"{context} ({layer_name})", layer_output_dir)

        # Save both accuracy and F1 results
        combined_results = {
            "accuracy": performance_results,
            "f1_scores": f1_results
        }
        
        with open(os.path.join(layer_output_dir, "accuracy_across_contexts.json"), "w") as f:
            f.write(json.dumps(performance_results))
            
        with open(os.path.join(layer_output_dir, "f1_scores_across_contexts.json"), "w") as f:
            f.write(json.dumps(f1_results))
            
        with open(os.path.join(layer_output_dir, "combined_metrics.json"), "w") as f:
            f.write(json.dumps(combined_results, indent=2))
            
        all_results[f"layer_{layer_idx}"] = combined_results
    
    # Save summary results if running multiple layers
    if len(layers_to_run) > 1:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "all_layers_summary.json"), "w") as f:
            f.write(json.dumps(all_results, indent=2))
        print(f"Summary results saved to {output_dir}/all_layers_summary.json")
    
    return all_results if len(layers_to_run) > 1 else performance_results
    


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear probe using pre-split or auto-generated splits.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", help="Name of the model to use from Hugging Face Hub.")
    parser.add_argument("--splits_dir", help="Directory containing pre-split train/val data")
    parser.add_argument("--dataset_path", help="Path to dataset JSON file (will auto-generate splits)")
    parser.add_argument("--output_dir", default="linear_probe_results", help="Directory to save the output JSON file.")
    parser.add_argument("--target_layers", type=int, nargs='+', default=None, help="The specific layer indices to extract the steering vector for. Can specify multiple layers. If not specified, runs on all layers.")
    parser.add_argument("--all_layers", action="store_true", help="Run linear probes on all layers in the model.")
    parser.add_argument("--num_epochs", type = int, default = 50, help = 'Number of epochs for linear probe training')
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate for linear probe training")
    
    random.seed(42)
    args = parser.parse_args()
    
    # Validate arguments
    if not args.splits_dir and not args.dataset_path:
        parser.error("Must provide either --splits_dir or --dataset_path")

    # Determine splits directory
    if args.splits_dir and os.path.exists(args.splits_dir):
        splits_dir = args.splits_dir
        print(f"Using existing splits from {splits_dir}")
    elif args.dataset_path:
        # Auto-generate splits directory name
        dataset_name = os.path.basename(args.dataset_path).replace('.json', '')
        splits_dir = f"{dataset_name}_splits"
        
        create_splits_if_needed(args.dataset_path, splits_dir)
        print(f"Using splits from {splits_dir}")
    else:
        raise ValueError("Splits directory not found and no dataset path provided")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.model_name, device)

    # Determine which layers to run
    if args.all_layers:
        target_layers = None  # Will run all layers
    elif args.target_layers is not None:
        target_layers = args.target_layers
    else:
        target_layers = None  # Default to all layers

    # Get contexts from splits directory
    contexts = get_contexts_from_splits(splits_dir)
    print(f"Found contexts: {sorted(contexts)}")
    
    # Store all results for cross-layer visualization
    all_context_results = {}
    
    for context in contexts:
        result = single_context(context, contexts, model, tokenizer, splits_dir, args, target_layers)
        all_context_results[context] = result
    
    # Generate cross-layer visualizations if running multiple layers
    if target_layers is None or (target_layers is not None and len(target_layers) > 1):
        print("Generating cross-layer performance visualizations...")
        
        # Create overall output directory for cross-layer plots
        cross_layer_output_dir = os.path.join(args.output_dir, "cross_layer_analysis")
        os.makedirs(cross_layer_output_dir, exist_ok=True)

        
        # Generate individual context plots showing accuracy and F1 across layers
        plot_individual_context_performance(all_context_results, cross_layer_output_dir, args.model_name)
        
        print(f"Individual context plots saved to {cross_layer_output_dir}")
    

if __name__ == "__main__":
    load_dotenv()
    main()


