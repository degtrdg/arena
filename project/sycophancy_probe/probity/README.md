# Probity

A Python library for creating, managing, and analyzing probes for large language models.

## Overview

Probity is a toolkit for interpretability research on neural networks, with a focus on analyzing internal representations through linear probing. It provides a comprehensive suite of tools for:

- Creating and managing datasets for probing experiments
- Collecting and storing model activations
- Training various types of probes (linear, logistic, PCA, etc.)
- Analyzing and interpreting probe results

### What Makes This Different From SKLearn?
- Runs on Pytorch natively where applicable
- Extensive dataset management tools, including tools for keeping track of character and token positions for labeled items
- Built-in activation collector, using TransformerLens as a model runner (will support NNsight in the near future as well)
- Designed for mech interp specifically

## Installation

```bash
# Clone the repository
git clone https://github.com/curt-tigges/probity.git
cd probity

# Install the library
pip install -e .
```

## Quick Start

```python
from probity.datasets.templated import TemplatedDataset, TemplateVariable, Template
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes.linear_probe import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from probity.probes.inference import ProbeInference
from transformers import AutoTokenizer

# Create a templated dataset
template_var = TemplateVariable(
    name="SENTIMENT", 
    values=["positive", "negative"],
    attributes={"label": [1, 0]},
    class_bound=True,  # This ensures this variable is always tied to the class 
    class_key="label"  
)

template = Template(
    template="This is a {SENTIMENT} example.",
    variables={"SENTIMENT": template_var},
    attributes={"task": "sentiment_analysis"}
)

dataset = TemplatedDataset(templates=[template])
probing_dataset = dataset.to_probing_dataset(
    label_from_attributes="label",
    auto_add_positions=True
)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=probing_dataset,
    tokenizer=tokenizer,
    padding=True,
    max_length=128,
    add_special_tokens=True
)

# Configure the probe, trainer, and pipeline
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
    name="sentiment_probe"
)

# Set up trainer configuration
trainer_config = SupervisedTrainerConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    weight_decay=0.01,
    train_ratio=0.8,  # 80-20 train-val split
    handle_class_imbalance=True,
    show_progress=True
)

# Set up pipeline configuration
pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=LogisticProbe,
    probe_config=probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=trainer_config,
    position_key="SENTIMENT",  # Probe at the sentiment word position
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/sentiment_probe_cache"
)

# Train the probe
pipeline = ProbePipeline(pipeline_config)
probe, training_history = pipeline.run()

# Use the probe for inference
inference = ProbeInference(
    model_name=model_name,
    hook_point=hook_point,
    probe=probe
)

texts = ["This is a positive example.", "This is a negative example."]
probabilities = inference.get_probabilities(texts)
```

## Key Features

### Dataset Creation

Probity offers multiple ways to create and manage datasets:

1. **Templated Datasets**: Create datasets with parametrized templates
2. **Custom Datasets**: Build datasets from scratch with fine-grained control
3. **Tokenization**: Convert character-based datasets to token-based for model compatibility
4. **Position Tracking**: Automatically track positions of interest in text

### Probe Types

The library supports various types of probes:

- **LinearProbe**: Simple linear probe with MSE loss
- **LogisticProbe**: Probe using logistic regression (binary classification)
- **MultiClassLogisticProbe**: Probe using logistic regression for multi-class classification
- **PCAProbe**: Probe using principal component analysis
- **KMeansProbe**: Probe using K-means clustering
- **MeanDifferenceProbe**: Probe using mean differences between classes

### Pipeline Components

- **Collection**: Tools for collecting and storing model activations
- **Training**: Supervised and unsupervised training frameworks
- **Inference**: Utilities for applying trained probes to new data
- **Analysis**: Methods for analyzing and visualizing probe results

## Examples

See the `tutorials/` directory for comprehensive examples of using Probity:

- `1-probity-basics.py`: Demonstrates the basic workflow with a Logistic Probe for sentiment analysis.
- `2-dataset-creation.py`: Shows various methods for creating templated and custom datasets.
- `3-probe-variants.py`: Compares different probe types (Linear, Logistic, PCA, KMeans, MeanDiff).
- `4-multiclass-probe.py`: Explains how to use the MultiClass Logistic Probe.

## Project Structure

- **probity/collection/**: Activation collection and storage
- **probity/datasets/**: Dataset creation and management
- **probity/probes/**: Probe implementation and utilities
- **probity/training/**: Training frameworks for probes
- **probity/utils/**: Utility functions and helpers
- **tutorials/**: Example notebooks and tutorials
- **tests/**: Unit and integration tests

## Citation

If you use Probity in your research, please cite:

```
@software{probity,
  author = {Tigges, Curt},
  title = {Probity: A Toolkit for Neural Network Probing},
  year = {2025},
  url = {https://github.com/curttigges/probity}
}
```