# Multi-View Capabilities

This repository contains a comprehensive framework for analyzing and steering the capabilities of large language models (LLMs). The project is designed to investigate how different "contexts" influence a model's behavior and to develop methods for controlling these behaviors using techniques like linear probing and activation steering.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Synthetic Data Generation](#1-synthetic-data-generation)
  - [2. Running Experiments](#2-running-experiments)
  - [3. Context Detection](#3-context-detection)
  - [4. Analysis and Visualization](#4-analysis-and-visualization)
- [Key Concepts](#key-concepts)
- [Contributing](#contributing)

## Project Overview

The core idea of this project is to understand and control LLM behaviors (referred to as "capabilities") by training linear probes on model activations. These probes can distinguish between different contexts and can be used to create "steering vectors" that guide the model's output.

The project is divided into several key components:

1.  **Synthetic Data Generation**: A pipeline for generating high-quality, labeled data for various capabilities and contexts using the Gemini API.
2.  **Experiment Pipeline**: A modular framework for running experiments, including training linear probes, generating steering vectors, and evaluating their effectiveness.
3.  **Context Detection**: A system for identifying the "context" of a given input using the trained linear probes.
4.  **Analysis and Visualization**: A suite of tools for analyzing experiment results and creating insightful visualizations.

## Features

-   **Synthetic Data Generation**: Automatically generate datasets for any capability and context.
-   **Linear Probing**: Train linear probes to identify specific behaviors in LLM activations.
-   **Activation Steering**: Use trained probes to create steering vectors that can control model behavior.
-   **Context Detection**: Detect the context of a given input to dynamically apply steering.
-   **Comprehensive Analysis**: Analyze the performance of probes and steering vectors across different layers, models, and contexts.
-   **Rich Visualizations**: Generate a variety of plots and charts to understand experimental results, including heatmaps, violin plots, and PCA analysis.
-   **Modular and Extensible**: The codebase is designed to be easily extended with new experiments, models, and analysis techniques.

## Directory Structure

```
.
├── context_detection/      # Scripts for context detection using trained probes
│   ├── analyze_context_detection.py
│   ├── context_detection.py
│   └── run_context_detection_pipeline.py
├── data/                   # Default directory for generated and cleaned datasets
├── experiment_pipeline/    # Core framework for running experiments
│   ├── analyze_probe_ratios.py
│   ├── experiments.py
│   ├── linear_probe.py
│   ├── steering_utils.py
│   └── visualize_probe_ratios.py
├── experiment1/            # Example experiment implementations
│   ├── steer_with_probe.py
│   ├── sv_with_bipo.py
│   └── eval.py
├── synth_data/             # Scripts for synthetic data generation
│   ├── generate.py
│   ├── data.py
│   └── deduplicate_questions.py
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/multi-view-capabilities.git
    cd multi-view-capabilities
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    This project does not include a `requirements.txt` file. You will need to install the following libraries:
    ```bash
    pip install torch transformers huggingface_hub scikit-learn matplotlib seaborn pandas python-dotenv google-generativeai
    ```

4.  **Set up API keys:**
    -   Create a `.env` file in the root directory.
    -   Add your Hugging Face and Gemini API keys to the `.env` file:
        ```
        HF_TOKEN="your_hugging_face_token"
        GEMINI_API_KEY="your_gemini_api_key"
        ```

## Usage

The project is designed to be run in a sequential manner, starting with data generation and moving through experimentation and analysis.

### 1. Synthetic Data Generation

The first step is to generate a synthetic dataset for the capabilities and contexts you want to study.

-   **Run the generation script:**
    ```bash
    python synth_data/generate.py
    ```
    The script will prompt you to enter a capability (e.g., "sycophancy") and the number of datapoints to generate. It will then use the Gemini API to generate a list of contexts, from which you can select the ones you want to use.

-   **Clean and deduplicate the data (optional but recommended):**
    ```bash
    python synth_data/data.py data/your_generated_data.json -o data/your_cleaned_data.json
    python synth_data/deduplicate_questions.py data/your_cleaned_data.json
    ```

### 2. Running Experiments

Once you have your dataset, you can run experiments to train linear probes and generate steering vectors.

-   **Train linear probes:**
    The `experiment_pipeline/linear_probe.py` script is the core of the experimentation process. It can be run as follows:
    ```bash
    python experiment_pipeline/linear_probe.py --model_name "Qwen/Qwen2.5-7B-Instruct" --dataset_path "data/your_dataset.json" --output_dir "results/your_experiment" --all_layers
    ```
    This will train a linear probe for each context in your dataset, across all layers of the specified model.

### 3. Context Detection

After training probes, you can use them to detect the context of new inputs.

-   **Run the context detection pipeline:**
    ```bash
    python context_detection/run_context_detection_pipeline.py --model_name "Qwen/Qwen2.5-7B-Instruct"
    ```
    This script will use the probes trained in the previous step to classify the context of various inputs.

### 4. Analysis and Visualization

The final step is to analyze the results of your experiments and create visualizations.

-   **Analyze probe accuracy ratios:**
    ```bash
    python experiment_pipeline/analyze_probe_ratios.py
    ```
    This script computes the ratio of general probe accuracy to context-specific probe accuracy and generates plots to visualize the results.

-   **Generate advanced visualizations:**
    ```bash
    python experiment_pipeline/visualize_probe_ratios.py
    ```
    This script creates more advanced visualizations, including heatmaps, violin plots, and PCA analysis, to help you understand the relationships between different capabilities and contexts.

## Key Concepts

-   **Capability**: A specific behavior or skill of an LLM, such as "refusal," "sycophancy," or "truthfulness."
-   **Context**: A specific situation or domain in which a capability is expressed. For example, the "refusal" capability could be expressed in the context of "illegal activities" or "personal opinions."
-   **Linear Probe**: A simple linear classifier trained on the internal activations of an LLM to detect the presence of a specific feature or behavior.
-   **Steering Vector**: A vector derived from a trained linear probe that can be added to the model's activations to encourage or discourage a specific behavior.
-   **Probe Accuracy Ratio**: The ratio of the accuracy of a general-purpose probe to the accuracy of a context-specific probe. This metric helps to determine how much a model's behavior is influenced by context.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.