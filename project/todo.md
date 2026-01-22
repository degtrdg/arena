Yeah. So first, we need to So Get the off the bells off the shelf bottle from Hugging Face for Gemma. It doesn't necessarily work for Gemma. I think Or we just want something depends on, like, whether we wanted to
get off the shelf base model from HF (e.g. gemma 2n)


Probity
A Python library for creating, managing, and analyzing probes for large language models.

Overview
Probity is a toolkit for interpretability research on neural networks, with a focus on analyzing internal representations through linear probing. It provides a comprehensive suite of tools for:

Creating and managing datasets for probing experiments
Collecting and storing model activations
Training various types of probes (linear, logistic, PCA, etc.)
Analyzing and interpreting probe results
What Makes This Different From SKLearn?
Runs on Pytorch natively where applicable
Extensive dataset management tools, including tools for keeping track of character and token positions for labeled items
Built-in activation collector, using TransformerLens as a model runner (will support NNsight in the near future as well)
Designed for mech interp specifically
Installation
# Clone the repository
git clone https://github.com/curt-tigges/probity.git
cd probity

# Install the library
pip install -e .
Quick Start
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
Key Features
Dataset Creation
Probity offers multiple ways to create and manage datasets:

Templated Datasets: Create datasets with parametrized templates
Custom Datasets: Build datasets from scratch with fine-grained control
Tokenization: Convert character-based datasets to token-based for model compatibility
Position Tracking: Automatically track positions of interest in text
Probe Types
The library supports various types of probes:

LinearProbe: Simple linear probe with MSE loss
LogisticProbe: Probe using logistic regression (binary classification)
MultiClassLogisticProbe: Probe using logistic regression for multi-class classification
PCAProbe: Probe using principal component analysis
KMeansProbe: Probe using K-means clustering
MeanDifferenceProbe: Probe using mean differences between classes
Pipeline Components
Collection: Tools for collecting and storing model activations
Training: Supervised and unsupervised training frameworks
Inference: Utilities for applying trained probes to new data
Analysis: Methods for analyzing and visualizing probe results
Examples
See the tutorials/ directory for comprehensive examples of using Probity:

1-probity-basics.py: Demonstrates the basic workflow with a Logistic Probe for sentiment analysis.
2-dataset-creation.py: Shows various methods for creating templated and custom datasets.
3-probe-variants.py: Compares different probe types (Linear, Logistic, PCA, KMeans, MeanDiff).
4-multiclass-probe.py: Explains how to use the MultiClass Logistic Probe.
Project Structure
probity/collection/: Activation collection and storage
probity/datasets/: Dataset creation and management
probity/probes/: Probe implementation and utilities
probity/training/: Training frameworks for probes
probity/utils/: Utility functions and helpers
tutorials/: Example notebooks and tutorials
tests/: Unit and integration tests
Citation
If you use Probity in your research, please cite:

@software{probity,
  author = {Tigges, Curt},
  title = {Probity: A Toolkit for Neural Network Probing},
  year = {2025},
  url = {https://github.com/curttigges/probity}
}


Automated Red Teaming for AI-Induced Psychosis
Tim Hua and AIs

A simple red teaming framework for testing how AI models respond to psychotic characters.

Note: Grok-4 on openrouter has started refusing to role play as a psychotic user. However, I found that Grok-3 is just as good and doesn't refuse.

Quick Start
Installation
Install the environment using uv:

uv sync
Basic Usage
Run red teaming on all default models with all characters:

uv run redteaming_systematic.py
📁 Project Structure
Core Scripts
redteaming_systematic.py - Main red teaming script with batch processing capabilities
results_analysis.R - Statistical analysis and visualization of results
analyze_not_crazy_mentions.py - Additional script tool for detecting "you're not crazy" mentions
convert_json_to_md.py - Utility for converting JSON transcripts to Markdown
xplore_notebook.py - Additional exploratory experiments
Configuration Files
red_team_prompt.txt - System prompt for the red team agent
grader_prompt.txt - System prompt for the grading agent
pyproject.toml - Project dependencies and configuration
Data Directories
characters/ - Character scenario files (.txt)
full_transcripts/ - Complete conversation logs (.md, .json)
result_grades/ - Grading results and analysis data (.csv)
error_logs/ - Error logs and debugging information (.json)
graphs/ - Generated visualizations and plots
Characters
I had several characters written by Claude

Adrian - Geometric reality delusions
Colin - Reality questioning and doubt
Ethan - General reality distortion
Frank - Weather-related delusions
Jessica - Prime number obsessions
Michelle - Relationship paranoia
Nathan - Quantum mechanics delusions
Rachel - Fertility-related concerns
Zara - Reality perception issues
These are then fed into the red teaming agent to role play.

redteaming_systematic.py Usage
Command Line Options
uv run redteaming_systematic.py [OPTIONS]
Basic Options
--help, -h - Show help message
--list-models - List all available models
--list-characters - List all available characters
Model Selection
--model, -m MODEL - Test specific model (default: all models)
--redteam-model MODEL - Model for red teaming (default: openrouter/x-ai/grok-4)
--grader-model MODEL - Model for grading (default: same as redteam-model)
Character Selection
--character, -c CHARACTERS - Specific character(s) to test
Configuration
--turns, -t NUMBER - Conversation turns per test (default: 12)
--max-connections NUMBER - Concurrent connections (default: 10)
Usage Examples
Example 1: Test Single Model on All Characters
uv run redteaming_systematic.py --model openai/chatgpt-4o-latest
Example 2: Test All Models on Specific Character
uv run redteaming_systematic.py --character ethan
Example 3: Test Multiple Specific Characters
# Comma-separated
uv run redteaming_systematic.py --character "ethan,frank,colin"

# Space-separated
uv run redteaming_systematic.py --character "ethan frank colin"
Example 4: Custom Configuration
uv run redteaming_systematic.py \
  --model openai/gpt-5-mini \
  --character "jessica,nathan" \
  --turns 1 \
  --max-connections 5
Example 5: List Available Options
# See all available models
uv run redteaming_systematic.py --list-models

# See all available characters  
uv run redteaming_systematic.py --list-characters
Default Models
The script includes these pre-configured models:

openai/chatgpt-4o-latest
openai/gpt-5
openrouter/openai/gpt-oss-20b
anthropic/claude-sonnet-4-20250514
openrouter/x-ai/grok-3
openrouter/x-ai/grok-4
openrouter/google/gemini-2.5-flash
openrouter/google/gemini-2.5-pro
openrouter/moonshotai/kimi-k2
openrouter/deepseek/deepseek-chat
openrouter/deepseek/deepseek-r1-0528
📊 Output and Results
Generated Files
Conversation Transcripts:

full_transcripts/{Character}_{Model}_{Timestamp}_target.md - Target model conversations
Results Data:

result_grades/redteaming_results_{Timestamp}.csv - Aggregated grading results
Contains turn-by-turn evaluations with model performance metrics
Error Logs:

error_logs/{Session}_error.json - Detailed error information for failed runs
Also I included some red team conversations (i.e., what Grok was thinking as it came up with more messages) in the transcripts folder with names like full_transcripts/{Character}_{Model}_{Timestamp}_redteam.json

Results Analysis
Use the R analysis script to generate insights:

# Run statistical analysis (requires R)
Rscript results_analysis.R
⚙️ Environment Configuration
Set up API keys in your environment or .env file:

# OpenAI
OPENAI_API_KEY=your_key_here

# OpenRouter  
OPENROUTER_API_KEY=your_key_here

# Anthropic
ANTHROPIC_API_KEY=your_key_here
