# %% [markdown]
# # Probe Type Comparison
# This file demonstrates how to:
# 1. Create a movie sentiment dataset
# 2. Train different types of probes (Linear, Logistic, KMeans, PCA, MeanDiff)
# 3. Run inference with each probe
# 4. Compare the learned directions using cosine similarity

# %% [markdown]
# ## Setup

# %% Setup and imports
import os
import torch
import numpy as np
import random
import torch.backends
import matplotlib.pyplot as plt
from tabulate import tabulate

from probity.datasets.templated import TemplatedDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from transformers import AutoTokenizer
from probity.probes import (
    LinearProbe,
    LinearProbeConfig,
    LogisticProbe,
    LogisticProbeConfig,
    KMeansProbe,
    KMeansProbeConfig,
    PCAProbe,
    PCAProbeConfig,
    MeanDifferenceProbe,
    MeanDiffProbeConfig,
)
from probity.training.trainer import (
    SupervisedProbeTrainer,
    SupervisedTrainerConfig,
    DirectionalProbeTrainer,
    DirectionalTrainerConfig,
)
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from probity.probes.inference import ProbeInference

# Set torch device consistently
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Set seed for reproducibility
set_seed(42)

# %% [markdown]
# ## Dataset Creation
# We'll use the same movie sentiment dataset as in 1-probity-basics.py for consistency.

# %%
# Create movie sentiment dataset
adjectives = {
    "positive": [
        "incredible",
        "amazing",
        "fantastic",
        "awesome",
        "beautiful",
        "brilliant",
        "exceptional",
        "extraordinary",
        "fabulous",
        "great",
        "lovely",
        "outstanding",
        "remarkable",
        "wonderful",
    ],
    "negative": [
        "terrible",
        "awful",
        "horrible",
        "bad",
        "disappointing",
        "disgusting",
        "dreadful",
        "horrendous",
        "mediocre",
        "miserable",
        "offensive",
        "terrible",
        "unpleasant",
        "wretched",
    ],
}
verbs = {
    "positive": ["loved", "enjoyed", "adored"],
    "negative": ["hated", "disliked", "detested"],
}

# Create dataset using factory method
movie_dataset = TemplatedDataset.from_movie_sentiment_template(
    adjectives=adjectives, verbs=verbs
)

# Convert to probing dataset with automatic position finding
# and label mapping from sentiment metadata
probing_dataset = movie_dataset.to_probing_dataset(
    label_from_attributes="sentiment",
    label_map={"positive": 1, "negative": 0},
    auto_add_positions=True,
)

# Convert to tokenized dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=probing_dataset,
    tokenizer=tokenizer,
    padding=True,  # Add padding
    max_length=128,  # Specify max length
    add_special_tokens=True,
)

# %% [markdown]
# ## Probe Training
# We'll train each type of probe and store their results.

# %%
# Common configuration
model_name = "gpt2"
hook_point = "blocks.7.hook_resid_pre"
hidden_size = 768  # GPT-2's hidden size

# Common trainer configuration for supervised probes
supervised_trainer_config = SupervisedTrainerConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    weight_decay=0.01,
    train_ratio=0.8,
    handle_class_imbalance=True,
    show_progress=True,
    device=device,
    standardize_activations=True,
)

# Common trainer configuration for directional probes
directional_trainer_config = DirectionalTrainerConfig(
    batch_size=32, device=device, standardize_activations=True
)

# Dictionary to store probes and their training histories
probes = {}
training_histories = {}

# %% [markdown]
# ### 1. Linear Probe
# A simple linear probe that learns a direction through gradient descent.

# %%
# Train Linear Probe
linear_probe_config = LinearProbeConfig(
    input_size=hidden_size,
    normalize_weights=True,
    bias=False,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=7,
    name="sentiment_linear_probe",
)

linear_pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=LinearProbe,
    probe_config=linear_probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=supervised_trainer_config,
    position_key="ADJ",
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/linear_probe_cache",
)

print("Training Linear Probe...")
linear_pipeline = ProbePipeline(linear_pipeline_config)
probe, history = linear_pipeline.run()
probes["linear"] = probe
training_histories["linear"] = history

# %% [markdown]
# ### 2. Logistic Probe
# A logistic regression probe that learns a direction for binary classification.

# %%
# Train Logistic Probe
logistic_probe_config = LogisticProbeConfig(
    input_size=hidden_size,
    normalize_weights=True,
    bias=False,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=7,
    name="sentiment_logistic_probe",
)

logistic_pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=LogisticProbe,
    probe_config=logistic_probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=supervised_trainer_config,
    position_key="ADJ",
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/logistic_probe_cache",
)

print("Training Logistic Probe...")
logistic_pipeline = ProbePipeline(logistic_pipeline_config)
probe, history = logistic_pipeline.run()
probes["logistic"] = probe
training_histories["logistic"] = history

# %% [markdown]
# ### 3. KMeans Probe
# A probe that finds directions through K-means clustering.

# %%
# Train KMeans Probe
kmeans_probe_config = KMeansProbeConfig(
    input_size=hidden_size,
    n_clusters=2,
    normalize_weights=True,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=7,
    name="sentiment_kmeans_probe",
)

kmeans_pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=KMeansProbe,
    probe_config=kmeans_probe_config,
    trainer_cls=DirectionalProbeTrainer,
    trainer_config=directional_trainer_config,
    position_key="ADJ",
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/kmeans_probe_cache",
)

print("Training KMeans Probe...")
kmeans_pipeline = ProbePipeline(kmeans_pipeline_config)
probe, history = kmeans_pipeline.run()
probes["kmeans"] = probe
training_histories["kmeans"] = history

# %% [markdown]
# ### 4. PCA Probe
# A probe that finds directions through principal component analysis.

# %%
# Train PCA Probe
pca_probe_config = PCAProbeConfig(
    input_size=hidden_size,
    n_components=1,
    normalize_weights=True,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=7,
    name="sentiment_pca_probe",
)

pca_pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=PCAProbe,
    probe_config=pca_probe_config,
    trainer_cls=DirectionalProbeTrainer,
    trainer_config=directional_trainer_config,
    position_key="ADJ",
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/pca_probe_cache",
)

print("Training PCA Probe...")
pca_pipeline = ProbePipeline(pca_pipeline_config)
probe, history = pca_pipeline.run()
probes["pca"] = probe
training_histories["pca"] = history

# %% [markdown]
# ### 5. Mean Difference Probe
# A probe that finds directions through mean differences between classes.

# %%
# Train Mean Difference Probe
meandiff_probe_config = MeanDiffProbeConfig(
    input_size=hidden_size,
    normalize_weights=True,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=7,
    name="sentiment_meandiff_probe",
)

meandiff_pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=MeanDifferenceProbe,
    probe_config=meandiff_probe_config,
    trainer_cls=DirectionalProbeTrainer,
    trainer_config=directional_trainer_config,
    position_key="ADJ",
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/meandiff_probe_cache",
)

print("Training Mean Difference Probe...")
meandiff_pipeline = ProbePipeline(meandiff_pipeline_config)
probe, history = meandiff_pipeline.run()
probes["meandiff"] = probe
training_histories["meandiff"] = history

# %%
# save all probes
save_dir = "./sentiment_probes"
os.makedirs(save_dir, exist_ok=True)
for probe_type, probe in probes.items():
    json_path = f"{save_dir}/{probe_type}_probe.json"
    probe.save_json(json_path)
    print(f"Saved probe JSON to {json_path}")

# %% [markdown]
# ## Training History Visualization
# Let's plot the training histories for the supervised probes (Linear and Logistic).

# %%
plt.figure(figsize=(12, 6))

# Plot training histories for supervised probes
for probe_type in ["linear", "logistic"]:
    history = training_histories[probe_type]
    plt.plot(history["train_loss"], label=f"{probe_type.title()} Train")
    plt.plot(history["val_loss"], label=f"{probe_type.title()} Val", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History for Supervised Probes")
plt.legend()
plt.show()

# %% [markdown]
# ## Inference Comparison
# Now let's compare how each probe performs on some test examples.

# %%
# Create test examples
test_examples = [
    "The movie was incredible and I loved every minute of it.",
    "That film was absolutely terrible and I hated it.",
    "The acting was mediocre but I liked the soundtrack.",
    "A beautiful story with outstanding performances.",
]

# Create inference objects for each probe
inference_objects = {}
for probe_type, probe in probes.items():
    inference_objects[probe_type] = ProbeInference(
        model_name=model_name, hook_point=hook_point, probe=probe, device=device
    )

# Get probabilities for each probe
results = {}
for probe_type, inference in inference_objects.items():
    results[probe_type] = inference.get_direction_activations(test_examples)

# Display results
print("\nInference Results:")
print("-" * 80)
for i, example in enumerate(test_examples):
    print(f"\nText: {example}")
    print("-" * 40)
    for probe_type, probs in results.items():
        # Get mean probability across all tokens as an overall sentiment score
        overall_sentiment = probs[i].mean().item()
        print(f"{probe_type.title()} Probe: {overall_sentiment:.4f}")

# %% [markdown]
# ## Direction Comparison
# Finally, let's compare the learned directions using cosine similarity.

# %%
# Get directions for each probe
directions = {}
for probe_type, probe in probes.items():
    directions[probe_type] = probe.get_direction()

# Create a matrix of cosine similarities
n_probes = len(directions)
similarity_matrix = torch.zeros((n_probes, n_probes))
probe_types = list(directions.keys())

for i, probe_type1 in enumerate(probe_types):
    for j, probe_type2 in enumerate(probe_types):
        # Compute cosine similarity between directions
        cos_sim = torch.nn.functional.cosine_similarity(
            directions[probe_type1], directions[probe_type2], dim=0
        )
        similarity_matrix[i, j] = cos_sim

# Display similarity matrix
print("\nDirection Similarity Matrix:")
print("-" * 80)

# Prepare data for tabulate
table_data = []
for i, probe_type1 in enumerate(probe_types):
    row = [probe_type1] + [f"{similarity_matrix[i, j]:.4f}" for j in range(n_probes)]
    table_data.append(row)

# Create headers
headers = ["Probe Type"] + probe_types

# Print table using tabulate
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# %% [markdown]
# ## Notes
# You'll see that the linear probe has the lowest cosine similarity to the other probes, though the cosine similarity is still meaningful given how
# high-dimensional the activation space is. It likely differs due to the different loss function (MSE vs. cross-entropy) which makes it a regression task
# instead of a classification task.

# %% [markdown]
# ## Conclusion
# This comparison shows how different probe types learn directions for the same task:
#
# 1. **Supervised Probes** (Linear and Logistic):
#    - Learn through gradient descent
#    - Can handle complex decision boundaries
#    - May take longer to train
#
# 2. **Directional Probes** (KMeans, PCA, MeanDiff):
#    - Learn through direct computation
#    - Faster to train
#    - May be more interpretable
#
# The cosine similarities between directions show how different methods may converge
# to similar or different solutions, just as shown in the *Language Models Linearly Represent Sentiment* paper. High similarities suggest the methods are
# finding consistent patterns in the activation space.
# %%
