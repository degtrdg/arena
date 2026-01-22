# %% [markdown]
# # Logistic Probe Basics
# This file demonstrates a complete workflow for:
# 1. Creating a movie sentiment dataset
# 2. Training a logistic regression probe
# 3. Running inference with the probe
# 4. Saving the probe to disk (in multiple formats)
# 5. Loading the probe back from disk
# 6. Verifying that the loaded probe gives consistent results

# %% [markdown]
# ## Setup

# %% Setup and imports
import torch
import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.backends

# Probity imports
from probity.datasets.templated import TemplatedDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.probes.inference import ProbeInference
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig

# Third-party imports
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from neuronpedia.np_vector import NPVector

# Set torch device consistently
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
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
# To train our probe, we'll create a dataset similar to the one used in the paper Linear Representations of Sentiment (Tigges & Hollinsworth, et al). We will use Probity's TemplatedDataset class, which allows us to specify templates with auto-populating blanks. For convenience, we have a simple function that applies the movie sentiment template.
#
# Once the TemplatedDataset is created, we simply convert it to a ProbingDataset (which gives us features like labels, word positions by character, and test-train splits) and then a TokenizedProbingDataset (which gives us additional information about token positions, context length, and so forth). We keep these distinct because TokenizedProbingDatasets are tied to specific models, whereas ProbingDatasets are not.

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

# Create dataset using multi-step approach
# Step 1: Create templated dataset
movie_dataset = TemplatedDataset.from_movie_sentiment_template(
    adjectives=adjectives, verbs=verbs
)

# Step 2: Convert to probing dataset with automatic position finding
# and label mapping from sentiment attributes
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

# %%
# Verify the tokenization worked
example = tokenized_dataset.examples[0]
print("First example tokens:", example.tokens)
print("First example text:", example.text)

# Now print examples from the probing dataset
print("Sample probing dataset examples:")
for i in np.random.choice(
    range(len(probing_dataset.examples)),
    size=min(6, len(probing_dataset.examples)),
    replace=False,
):
    ex = probing_dataset.examples[i]
    label = "positive" if ex.label == 1 else "negative"
    print(f"Example {i}: '{ex.text}' (Label: {label})")

# %% [markdown]
# ## Probe Training
# ### Configuration
# We're now ready to train the probe! We specify the training parameters via the three config objects below. The Pipeline manages the whole process, and the Trainer and Probe have their own settings.

# %% Configure model and probe
model_name = "gpt2"
hook_point = "blocks.7.hook_resid_pre"

# Set up logistic probe configuration
probe_config = LogisticProbeConfig(
    input_size=768,
    normalize_weights=True,
    bias=False,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=7,
    name="sentiment_probe",
)

# Set up trainer configuration
trainer_config = SupervisedTrainerConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    weight_decay=0.01,
    train_ratio=0.8,  # 80-20 train-val split
    handle_class_imbalance=True,
    show_progress=True,
    standardize_activations=True,  # Empirically, this produces better results
)

pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=LogisticProbe,
    probe_config=probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=trainer_config,
    position_key="ADJ",  # We want to probe at the adjective position
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/sentiment_probe_cache",  # Cache activations for reuse
)

# %%
# Let's make sure the position key is correct
model = HookedTransformer.from_pretrained(model_name)

example = tokenized_dataset.examples[0]
print(f"Example text: {model.to_str_tokens(example.text, prepend_bos=False)}")
print(f"Token positions: {example.token_positions}")
print(f"Available position keys: {list(example.token_positions.keys())}")

# Verify the position key matches what's in the dataset
print(f"\nPipeline position key: {pipeline_config.position_key}")

# %% [markdown]
# Looks like the key tokens are in the right positions. GPT2's default behavior (as implemented in the AutoTokenizer) is not to add a BOS, so we're fine in that respect.
#
# ### Training
# Let's train the probe!

# %% Collect activations
# Create and run pipeline
pipeline = ProbePipeline(pipeline_config)

probe, training_history = pipeline.run()

# The probe now contains our learned sentiment direction
sentiment_direction = probe.get_direction()

# %%
# We can analyze training history
plt.figure(figsize=(10, 5))
plt.plot(training_history["train_loss"], label="Train Loss")
plt.plot(training_history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Probe Training History")
plt.legend()
plt.show()

# %% [markdown]
# # Now we'll test the standardized inference path
#
# ## Understanding how inference differs from training
#
# When running inference with a trained probe, it's important to understand:
#
# 1. **Position keys are only used during training** - While we specified `position_key="ADJ"` during training
#    to learn the sentiment direction specifically from adjective positions, the trained probe doesn't "remember"
#    this position information during inference.
#
# 2. **Inference applies to all tokens** - The probe is applied to every token in the input text, giving
#    a score for each token position. This means even function words like "the" and "and" get scores.
#
# 3. **Format independence** - You can run inference on any text format, not just the template used for training.
#    However, the probe will be most reliable when applied to similar linguistic patterns as seen during training.


# %%
# Method 1: Inference with the trained probe object
print("===== Inference with trained probe object =====")

# Create a ProbeInference instance with the trained probe
inference = ProbeInference(
    model_name=model_name, hook_point=hook_point, probe=probe, device=device
)

# Create some test examples with different formats than our training template
test_examples = [
    "The movie was incredible and I loved every minute of it.",  # Similar to training format
    "That film was absolutely terrible and I hated it.",  # Similar format with synonyms
    "The acting was mediocre but I liked the soundtrack.",  # Mixed sentiment
    "A beautiful story with outstanding performances.",  # Different format
]

# Get raw activation scores along the probe direction
raw_scores = inference.get_direction_activations(test_examples)
print(f"Raw activation scores shape: {raw_scores.shape}")

# Get probabilities (applies sigmoid for logistic probes)
probabilities = inference.get_probabilities(test_examples)
print(f"Probabilities shape: {probabilities.shape}")

# Display the results with token-level breakdown
for i, example in enumerate(test_examples):
    print(f"\nText: {example}")

    # Show token-level analysis
    tokens = inference.model.to_str_tokens(example, prepend_bos=False)
    token_probs = probabilities[i]

    # Make sure we only process as many tokens as we have probabilities for
    max_tokens = min(len(tokens), len(token_probs) - 1)

    print("\nToken-level sentiment analysis:")
    print(f"{'Token':<15} {'Sentiment Score':<15} {'Interpretation'}")
    print("-" * 50)

    for j in range(max_tokens):
        token = tokens[j]

        score = token_probs[j].item()
        interp = "Positive" if score > 0.6 else "Negative" if score < 0.1 else "Neutral"
        print(f"{token:<15} {score:.4f}           {interp}")

    # Get mean probability across tokens (excluding BOS) as sentiment score
    overall_sentiment = token_probs[1 : max_tokens + 1].mean().item()
    print(
        f"\nOverall document sentiment: {overall_sentiment:.4f} ({'Positive' if overall_sentiment > 0.6 else 'Negative' if overall_sentiment < 0.4 else 'Neutral'})"
    )

# %% [markdown]
# ## Observations on the token-level scores
#
# Looking at the token-level breakdown:
#
# 1. **Content words matter most**: Words like "incredible", "terrible", "loved", and "hated" show
#    the strongest sentiment signals, while function words like "the" and "and" are more neutral.
#
# 2. **Contextual influence**: Even seemingly neutral words can be affected by surrounding context.
#    The same word might get different scores in different sentences.
#
# 3. **Format robustness**: The probe works well on sentences with different structures than our training
#    template, showing it captures generalizable sentiment features.
#
# 4. **Aggregation strategy**: Taking the mean across all tokens is simple but not always optimal.
#    In practice, you might want to weight content words more heavily or focus on specific parts of speech.

# %% [markdown]
# # Save the probe and load it back

# %%
# Save the probe in both formats
save_dir = "./saved_probes"
os.makedirs(save_dir, exist_ok=True)

# Option 1: Save as PyTorch model (full state and config)
probe_path = f"{save_dir}/sentiment_probe.pt"
probe.save(probe_path)
print(f"Saved probe to {probe_path}")

# Option 2: Save in JSON format for easier sharing
json_path = f"{save_dir}/sentiment_probe.json"
probe.save_json(json_path)
print(f"Saved probe JSON to {json_path}")

# %%
# Method 2: Inference with the saved probe
print("\n===== Inference with loaded probe =====")

# Load the probe using from_saved_probe
loaded_inference = ProbeInference.from_saved_probe(
    model_name=model_name,
    hook_point=hook_point,
    probe_path=json_path,  # Load from the JSON format
    device=device,
)

# Get results with loaded probe
loaded_probabilities = loaded_inference.get_probabilities(test_examples)

# Compare results between original and loaded probes
for i, example in enumerate(test_examples):
    print(f"\nText: {example}")

    # Show token-level analysis
    tokens = loaded_inference.model.to_str_tokens(example, prepend_bos=False)
    token_probs = loaded_probabilities[i]

    # Make sure we only process as many tokens as we have probabilities for
    # Skip the BOS token in the probabilities
    max_tokens = min(len(tokens), len(token_probs) - 1)

    # Get document-level sentiment score (excluding BOS)
    overall_sentiment = token_probs[1 : max_tokens + 1].mean().item()
    print(
        f"Overall document sentiment (loaded probe): {overall_sentiment:.4f} ({'Positive' if overall_sentiment > 0.6 else 'Negative' if overall_sentiment < 0.4 else 'Neutral'})"
    )

# %% [markdown]
# # Verify consistency between original and loaded probes

# %%
# Compare original and loaded probe results
print("\n===== Comparing original vs loaded probe results =====")

# Get the probe directions
original_direction = probe.get_direction()
loaded_direction = loaded_inference.probe.get_direction()

# Print direction norm and first few components
print(f"Original direction norm: {torch.norm(original_direction):.6f}")
print(f"Original direction [0:5]: {original_direction[:5].cpu().tolist()}")

print(f"Loaded direction norm: {torch.norm(loaded_direction):.6f}")
print(f"Loaded direction [0:5]: {loaded_direction[:5].cpu().tolist()}")

# Check if the directions are similar
cos_sim = torch.nn.functional.cosine_similarity(
    original_direction, loaded_direction, dim=0
)
print(f"Cosine similarity between directions: {cos_sim.item():.6f}")

# Get new raw scores for comparison if we need them
raw_scores = inference.get_direction_activations(test_examples)
loaded_raw_scores = loaded_inference.get_direction_activations(test_examples)

# Analyze where the differences are greatest
token_diffs = torch.abs(raw_scores - loaded_raw_scores).mean(dim=0)
print("\nToken-level mean absolute differences:")
for i, diff in enumerate(token_diffs):
    print(f"Token {i}: {diff.item():.6f}")

max_diff_token = torch.argmax(token_diffs).item()
print(f"\nLargest difference at token position {max_diff_token}")

# Check difference excluding the BOS token
non_bos_raw_diff = torch.abs(raw_scores[:, 1:] - loaded_raw_scores[:, 1:]).mean().item()
non_bos_prob_diff = (
    torch.abs(probabilities[:, 1:] - loaded_probabilities[:, 1:]).mean().item()
)
print(f"Mean absolute difference in raw scores (excluding BOS): {non_bos_raw_diff:.8f}")
print(
    f"Mean absolute difference in probabilities (excluding BOS): {non_bos_prob_diff:.8f}"
)

if non_bos_prob_diff < 0.3:
    print("The non-BOS tokens show reasonable consistency.")

# Verify the trend is similar (correlation between original and loaded results)
flattened_orig = probabilities[:, 1:].flatten()
flattened_loaded = loaded_probabilities[:, 1:].flatten()
correlation = torch.corrcoef(torch.stack([flattened_orig, flattened_loaded]))[
    0, 1
].item()
print(f"Correlation between original and loaded probe outputs: {correlation:.6f}")

if correlation > 0.8:
    print("SUCCESS: Original and loaded probes show strong correlation (same trend)!")
elif correlation > 0.5:
    print("PARTIAL SUCCESS: Original and loaded probes show moderate correlation.")

# %% [markdown]
# # Customizing token selection for arbitrary text
#
# While probes are applied to all tokens during inference, you often want to focus on specific tokens
# for different text formats. Here's a pattern for selecting tokens of interest:


# %%
def analyze_with_focus(text, inference, focus_words=None, pos_tags=None):
    """Analyze text with focus on specific words or POS tags."""
    print(f"\nAnalyzing: {text}")

    # Get probabilities for all tokens
    probs = inference.get_probabilities([text])[0]
    tokens = inference.model.to_str_tokens(text, prepend_bos=False)

    # Handle token count mismatch - skip BOS token in probabilities
    max_tokens = min(len(tokens), len(probs) - 1)

    # Print all token scores
    print("\nAll token scores:")
    for i in range(max_tokens):
        token = tokens[i]
        score = probs[i].item()
        print(f"{token:<12} {score:.4f}")

    # Focus on specific words if requested
    if focus_words:
        print("\nFocusing on specific words:")
        focus_indices = [
            i
            for i in range(max_tokens)
            if any(word in tokens[i].lower() for word in focus_words)
        ]
        if focus_indices:
            focus_tokens = [tokens[i] for i in focus_indices]
            adjusted_indices = [i for i in focus_indices]
            focus_scores = probs[adjusted_indices].mean().item()
            print(f"Words: {', '.join(focus_tokens)}")
            print(f"Average sentiment: {focus_scores:.4f}")
        else:
            print("No matching words found")

    # Calculate overall sentiment
    avg_score = probs[1:max_tokens].mean().item()
    sentiment = (
        "Positive" if avg_score > 0.55 else "Negative" if avg_score < 0.4 else "Neutral"
    )
    print(f"\nOverall sentiment: {avg_score:.4f} ({sentiment})")

    return avg_score


# Try with some very different text formats
complex_review = "Despite excellent visuals and strong performances from the lead actors, the plot was incoherent and the pacing dragged in the middle."

# Analyze with focus on content words related to different aspects
print("\n===== Customized token selection =====")
analyze_with_focus(
    complex_review, inference, focus_words=["visual", "perform", "actor"]
)
analyze_with_focus(complex_review, inference, focus_words=["plot", "pac", "middle"])


# %% [markdown]
# # Uploading to Neuronpedia
#
# Once you've trained a probe, you can upload it to Neuronpedia. The process is straightforward:
# 1. Get the probe direction
# 2. Set your API key
# 3. Upload the probe
# 4. Profit

# %%
# Get the probe direction
probe_direction = probe.get_direction()

# Convert to list of floats
probe_direction = probe_direction.tolist()

from neuronpedia.np_vector import NPVector

# from neuronpedia.org/account
os.environ["NEURONPEDIA_API_KEY"] = "YOUR_API_KEY"

# upload the custom vector
np_vector = NPVector.new(
    label="sentiment_probe",
    model_id="gpt2-small",
    layer_num=7,
    hook_type="hook_resid_pre",
    vector=probe_direction,
    default_steer_strength=20,
)

# steer with it
responseJson = np_vector.steer_completion(prompt="The movie was")

print(json.dumps(responseJson, indent=2))
print("UI Steering at: " + responseJson["shareUrl"])

# %% [markdown]
# # Conclusion
#
# - **The preferred workflow is**:
#    - Train a probe (using a trainer or pipeline)
#    - Save it with probe.save() or probe.save_json()
#    - Load it with ProbeInference.from_saved_probe()
#    - Use get_direction_activations() for raw scores or get_probabilities() for transformed outputs
