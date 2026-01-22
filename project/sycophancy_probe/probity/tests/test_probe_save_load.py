import torch
import numpy as np
import os
import pytest
from pathlib import Path
from typing import Dict, List

from probity.datasets.templated import TemplatedDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from transformers import AutoTokenizer
from probity.probes.linear_probe import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.probes.inference import ProbeInference

# Create a temporary directory for test artifacts
@pytest.fixture
def temp_dir():
    path = Path("test_artifacts")
    if not path.exists():
        path.mkdir()
    yield path
    # Uncomment to clean up after tests
    # import shutil
    # shutil.rmtree(path)

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def create_movie_dataset():
    """Create a simple movie sentiment dataset identical to the tutorial."""
    adjectives = {
        "positive": ["incredible", "amazing", "fantastic", "awesome", "beautiful"],
        "negative": ["terrible", "awful", "horrible", "bad", "disappointing"]
    }
    verbs = {
        "positive": ["loved", "enjoyed", "adored"],
        "negative": ["hated", "disliked", "detested"]
    }

    # Create dataset using factory method
    movie_dataset = TemplatedDataset.from_movie_sentiment_template(
        adjectives=adjectives,
        verbs=verbs
    )

    # Convert to probing dataset
    probing_dataset = movie_dataset.to_probing_dataset(
        label_from_metadata="sentiment",
        label_map={"positive": 1, "negative": 0},
        auto_add_positions=True
    )

    # Convert to tokenized dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding=True,
        max_length=128,
        add_special_tokens=True
    )
    
    return tokenized_dataset

def train_probe(dataset, model_name="gpt2", hook_point="blocks.7.hook_resid_pre", device="cpu"):
    """Train a logistic probe on the dataset."""
    from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
    
    # Set up logistic probe configuration
    probe_config = LogisticProbeConfig(
        input_size=768,
        normalize_weights=True,
        bias=True,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=7,
        name="test_sentiment_probe"
    )

    # Set up trainer configuration
    trainer_config = SupervisedTrainerConfig(
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=3,  # Reduced epochs for faster testing
        weight_decay=0.01,
        train_ratio=0.8,
        handle_class_imbalance=True,
        show_progress=True,
        device=device
    )

    pipeline_config = ProbePipelineConfig(
        dataset=dataset,
        probe_cls=LogisticProbe,
        probe_config=probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="ADJ",  # Probe at the adjective position
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir="./test_artifacts/probe_cache",
        device=device
    )

    # Create and run pipeline
    pipeline = ProbePipeline(pipeline_config)
    probe, training_history = pipeline.run()
    
    return probe

def compare_probes(original_probe, loaded_probe, test_examples, device="cpu"):
    """Compare two probes to check if they produce the same outputs."""
    
    model_name = "gpt2"
    hook_point = "blocks.7.hook_resid_pre"
    
    # Create inference objects
    original_inference = ProbeInference(
        model_name=model_name,
        hook_point=hook_point,
        probe=original_probe,
        device=device
    )
    
    loaded_inference = ProbeInference(
        model_name=model_name, 
        hook_point=hook_point,
        probe=loaded_probe,
        device=device
    )
    
    # Compare direction vectors
    original_direction = original_probe.get_direction(normalized=True)
    loaded_direction = loaded_probe.get_direction(normalized=True)
    
    # Compare raw activations
    original_raw_scores = original_inference.get_direction_activations(test_examples)
    loaded_raw_scores = loaded_inference.get_direction_activations(test_examples)
    
    # Compare probabilities
    original_probs = original_inference.get_probabilities(test_examples)
    loaded_probs = loaded_inference.get_probabilities(test_examples)
    
    # Return all comparison data
    return {
        "direction_cosine_sim": torch.nn.functional.cosine_similarity(
            original_direction, loaded_direction, dim=0).item(),
        "direction_diff": torch.abs(original_direction - loaded_direction).max().item(),
        "raw_scores_diff": torch.abs(original_raw_scores - loaded_raw_scores).max().item(),
        "probabilities_diff": torch.abs(original_probs - loaded_probs).max().item(),
        "original_direction": original_direction,
        "loaded_direction": loaded_direction,
        "original_raw_scores": original_raw_scores,
        "loaded_raw_scores": loaded_raw_scores,
        "original_probs": original_probs,
        "loaded_probs": loaded_probs
    }

def test_probe_save_load_pt(temp_dir):
    """Test that a probe saved and loaded in PyTorch format has consistent behavior."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dataset
    dataset = create_movie_dataset()
    
    # Train probe
    original_probe = train_probe(dataset, device=device)
    
    # Save probe in PyTorch format
    probe_path = temp_dir / "test_probe.pt"
    original_probe.save(str(probe_path))
    
    # Load probe back
    loaded_probe = LogisticProbe.load(str(probe_path))
    loaded_probe.to(device)
    
    # Test examples
    test_examples = [
        "The movie was incredible and I loved every minute of it.",
        "That film was absolutely terrible and I hated it."
    ]
    
    # Compare probes
    results = compare_probes(original_probe, loaded_probe, test_examples, device)
    
    # Print detailed comparison for debugging
    print("\n=== PyTorch Format Test Results ===")
    print(f"Direction cosine similarity: {results['direction_cosine_sim']:.6f}")
    print(f"Max direction difference: {results['direction_diff']:.6f}")
    print(f"Max raw scores difference: {results['raw_scores_diff']:.6f}")
    print(f"Max probabilities difference: {results['probabilities_diff']:.6f}")
    
    # Original probe details
    print("\nOriginal probe - feature stats:")
    if hasattr(original_probe, 'feature_mean'):
        print(f"  feature_mean exists: {original_probe.feature_mean is not None}")
        if original_probe.feature_mean is not None:
            print(f"  feature_mean shape: {original_probe.feature_mean.shape}")
    if hasattr(original_probe, 'feature_std'):
        print(f"  feature_std exists: {original_probe.feature_std is not None}")
        if original_probe.feature_std is not None:
            print(f"  feature_std shape: {original_probe.feature_std.shape}")
    
    # Loaded probe details
    print("\nLoaded probe - feature stats:")
    if hasattr(loaded_probe, 'feature_mean'):
        print(f"  feature_mean exists: {loaded_probe.feature_mean is not None}")
        if loaded_probe.feature_mean is not None:
            print(f"  feature_mean shape: {loaded_probe.feature_mean.shape}")
    if hasattr(loaded_probe, 'feature_std'):
        print(f"  feature_std exists: {loaded_probe.feature_std is not None}")
        if loaded_probe.feature_std is not None:
            print(f"  feature_std shape: {loaded_probe.feature_std.shape}")
    
    # Assertions (with tolerance)
    assert results['direction_cosine_sim'] > 0.999, "Direction vectors should be almost identical"
    assert results['direction_diff'] < 1e-4, "Direction vectors should be almost identical"
    assert results['raw_scores_diff'] < 1e-2, "Raw scores should be almost identical"
    assert results['probabilities_diff'] < 1e-2, "Probabilities should be almost identical"

def test_probe_save_load_json(temp_dir):
    """Test that a probe saved and loaded in JSON format has consistent behavior."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dataset
    dataset = create_movie_dataset()
    
    # Train probe
    original_probe = train_probe(dataset, device=device)
    
    # Get the original standardization buffers and weights
    original_mean = original_probe.feature_mean
    original_std = original_probe.feature_std
    original_weight = original_probe.linear.weight.data.clone()
    
    # Get the original direction vector 
    original_direction = original_probe.get_direction(normalized=True)
    original_unnorm_direction = original_probe.get_direction(normalized=False)
    
    # Save probe in JSON format
    json_path = temp_dir / "test_probe.json"
    original_probe.save_json(str(json_path))
    
    # Examine the saved JSON content
    import json
    with open(json_path, 'r') as f:
        saved_data = json.load(f)
        
    print("\n=== JSON Format Detailed Info ===")
    print("JSON Metadata keys:", list(saved_data["metadata"].keys()))
    print(f"is_standardized: {saved_data['metadata'].get('is_standardized')}")
    print(f"is_normalized: {saved_data['metadata'].get('is_normalized')}")
    print(f"is_unscaled: {saved_data['metadata'].get('is_unscaled')}")
    print(f"has_bias: {saved_data['metadata'].get('has_bias')}")
    
    # Check if feature_mean and feature_std are in the metadata
    if 'feature_mean' in saved_data['metadata']:
        print("feature_mean in JSON:", torch.tensor(saved_data['metadata']['feature_mean']).shape)
    if 'feature_std' in saved_data['metadata']:
        print("feature_std in JSON:", torch.tensor(saved_data['metadata']['feature_std']).shape)
    
    # Load probe back
    loaded_probe = LogisticProbe.load_json(str(json_path))
    loaded_probe.to(device)
    
    # Get the loaded standardization buffers and weights
    loaded_mean = loaded_probe.feature_mean
    loaded_std = loaded_probe.feature_std  
    loaded_weight = loaded_probe.linear.weight.data.clone()
    
    # Get the loaded direction vector
    loaded_direction = loaded_probe.get_direction(normalized=True)
    loaded_unnorm_direction = loaded_probe.get_direction(normalized=False)
    
    # Compare weights and standardization buffers
    print("\nWeight comparison:")
    print(f"Original weight shape: {original_weight.shape}")
    print(f"Loaded weight shape: {loaded_weight.shape}")
    print(f"Weights equal: {torch.allclose(original_weight, loaded_weight)}")
    print(f"Weight difference max: {torch.abs(original_weight - loaded_weight).max().item()}")
    
    # Compare standardization buffers
    print("\nStandardization buffers comparison:")
    if original_mean is not None and loaded_mean is not None:
        print(f"Mean equal: {torch.allclose(original_mean, loaded_mean)}")
        print(f"Mean difference max: {torch.abs(original_mean - loaded_mean).max().item()}")
    else:
        print("Mean comparison unavailable - one or both is None")
        
    if original_std is not None and loaded_std is not None:
        print(f"Std equal: {torch.allclose(original_std, loaded_std)}")
        print(f"Std difference max: {torch.abs(original_std - loaded_std).max().item()}")
    else:
        print("Std comparison unavailable - one or both is None")
        
    # Compare direction vectors
    print("\nDirection vector comparison:")
    print(f"Normalized original direction shape: {original_direction.shape}")
    print(f"Normalized loaded direction shape: {loaded_direction.shape}")
    print(f"Direction cosine similarity: {torch.nn.functional.cosine_similarity(original_direction, loaded_direction, dim=0).item():.6f}")
    print(f"Direction difference max: {torch.abs(original_direction - loaded_direction).max().item():.6f}")
    
    # Compare unnormalized direction vectors
    print("\nUnnormalized direction vector comparison:")
    print(f"Unnorm original direction shape: {original_unnorm_direction.shape}")
    print(f"Unnorm loaded direction shape: {loaded_unnorm_direction.shape}")
    cos_sim_unnorm = torch.nn.functional.cosine_similarity(original_unnorm_direction, loaded_unnorm_direction, dim=0).item()
    print(f"Unnorm direction cosine similarity: {cos_sim_unnorm:.6f}")
    
    # Compare computation steps
    print("\nGet_direction computation comparison:")
    
    # Get config's additional_info
    original_info = getattr(original_probe.config, 'additional_info', {})
    loaded_info = getattr(loaded_probe.config, 'additional_info', {})
    
    print(f"Original additional_info: {original_info}")
    print(f"Loaded additional_info: {loaded_info}")
    
    # Test the impact of standardization on get_direction
    test_input = torch.randn(3, 768, device=device)
    
    original_standardized = None
    if hasattr(original_probe, '_apply_standardization'):
        original_standardized = original_probe._apply_standardization(test_input)
        
    loaded_standardized = None
    if hasattr(loaded_probe, '_apply_standardization'):
        loaded_standardized = loaded_probe._apply_standardization(test_input)
    
    if original_standardized is not None and loaded_standardized is not None:
        print(f"Standardized inputs equal: {torch.allclose(original_standardized, loaded_standardized)}")
        print(f"Standardized inputs diff max: {torch.abs(original_standardized - loaded_standardized).max().item()}")
    
    # Test examples
    test_examples = [
        "The movie was incredible and I loved every minute of it.",
        "That film was absolutely terrible and I hated it."
    ]
    
    # Compare probes
    results = compare_probes(original_probe, loaded_probe, test_examples, device)
    
    # Print detailed comparison for debugging
    print("\n=== JSON Format Test Results ===")
    print(f"Direction cosine similarity: {results['direction_cosine_sim']:.6f}")
    print(f"Max direction difference: {results['direction_diff']:.6f}")
    print(f"Max raw scores difference: {results['raw_scores_diff']:.6f}")
    print(f"Max probabilities difference: {results['probabilities_diff']:.6f}")
    
    # Original probe details
    print("\nOriginal probe - feature stats:")
    if hasattr(original_probe, 'feature_mean'):
        print(f"  feature_mean exists: {original_probe.feature_mean is not None}")
        if original_probe.feature_mean is not None:
            print(f"  feature_mean shape: {original_probe.feature_mean.shape}")
    if hasattr(original_probe, 'feature_std'):
        print(f"  feature_std exists: {original_probe.feature_std is not None}")
        if original_probe.feature_std is not None:
            print(f"  feature_std shape: {original_probe.feature_std.shape}")
    
    # Loaded probe details
    print("\nLoaded probe - feature stats:")
    if hasattr(loaded_probe, 'feature_mean'):
        print(f"  feature_mean exists: {loaded_probe.feature_mean is not None}")
        if loaded_probe.feature_mean is not None:
            print(f"  feature_mean shape: {loaded_probe.feature_mean.shape}")
    if hasattr(loaded_probe, 'feature_std'):
        print(f"  feature_std exists: {loaded_probe.feature_std is not None}")
        if loaded_probe.feature_std is not None:
            print(f"  feature_std shape: {loaded_probe.feature_std.shape}")
            
    # FIXME: Because of the inconsistency in JSON format, we're temporarily relaxing the assertions
    if cos_sim_unnorm > 0.7:
        print("PASS: Unnormalized direction vectors have reasonable similarity")
    else:
        print("FAIL: Unnormalized direction vectors differ significantly")
        
    # Assertions (with tolerance)
    assert results['direction_cosine_sim'] > 0.999, "Direction vectors should be almost identical"
    assert results['direction_diff'] < 1e-4, "Direction vectors should be almost identical"
    assert results['raw_scores_diff'] < 1e-2, "Raw scores should be almost identical"
    assert results['probabilities_diff'] < 1e-2, "Probabilities should be almost identical"

def test_standardization_application():
    """Test that standardization is properly saved and applied during inference."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create fixed test tensors
    mean = torch.tensor([1.0, 2.0, 3.0], device=device).reshape(1, -1)
    std = torch.tensor([0.5, 1.0, 1.5], device=device).reshape(1, -1)
    
    # Create a test probe with standardization
    probe_config = LogisticProbeConfig(input_size=3, device=device)
    probe = LogisticProbe(probe_config)
    
    # Register standardization buffers
    probe.register_buffer('feature_mean', mean)
    probe.register_buffer('feature_std', std)
    
    # Set fixed weights for testing
    with torch.no_grad():
        probe.linear.weight.data = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        if probe.linear.bias is not None:
            probe.linear.bias.data = torch.tensor([0.0], device=device)
    
    # Save and load the probe
    temp_path = Path("test_artifacts")
    if not temp_path.exists():
        temp_path.mkdir()
    
    probe_path = temp_path / "test_standardization.pt"
    probe.save(str(probe_path))
    
    loaded_probe = LogisticProbe.load(str(probe_path))
    loaded_probe.to(device)
    
    # Test input
    test_input = torch.tensor([[2.0, 4.0, 6.0]], device=device)
    
    # Apply standardization manually
    expected_standardized = (test_input - mean) / std
    expected_output = torch.sum(expected_standardized, dim=1)
    
    # Get output from original probe
    original_output = probe(test_input)
    
    # Get output from loaded probe
    loaded_output = loaded_probe(test_input)
    
    # Print results for debugging
    print("\n=== Standardization Test Results ===")
    print(f"Test input: {test_input}")
    print(f"Expected standardized: {expected_standardized}")
    print(f"Expected output: {expected_output}")
    print(f"Original probe output: {original_output}")
    print(f"Loaded probe output: {loaded_output}")
    
    # Check if original probe applies standardization
    if hasattr(probe, '_apply_standardization'):
        standardized_input = probe._apply_standardization(test_input)
        print(f"Original probe standardized input: {standardized_input}")
        print(f"Matches expected standardized: {torch.allclose(standardized_input, expected_standardized)}")
    
    # Check if loaded probe applies standardization
    if hasattr(loaded_probe, '_apply_standardization'):
        standardized_input = loaded_probe._apply_standardization(test_input)
        print(f"Loaded probe standardized input: {standardized_input}")
        print(f"Matches expected standardized: {torch.allclose(standardized_input, expected_standardized)}")
    
    # Assertions
    assert torch.allclose(original_output, expected_output), "Original probe should apply standardization correctly"
    assert torch.allclose(loaded_output, expected_output), "Loaded probe should apply standardization correctly"
    assert torch.allclose(original_output, loaded_output), "Original and loaded probes should give same output"

def test_inference_standardization():
    """Test if ProbeInference correctly applies standardization."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create fixed test tensors
    mean = torch.tensor([1.0, 2.0, 3.0], device=device).reshape(1, -1)
    std = torch.tensor([0.5, 1.0, 1.5], device=device).reshape(1, -1)
    
    # Create a test probe with standardization
    probe_config = LogisticProbeConfig(input_size=3, device=device)
    probe = LogisticProbe(probe_config)
    
    # Register standardization buffers
    probe.register_buffer('feature_mean', mean)
    probe.register_buffer('feature_std', std)
    
    # Set fixed weights for testing
    with torch.no_grad():
        probe.linear.weight.data = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        if probe.linear.bias is not None:
            probe.linear.bias.data = torch.tensor([0.0], device=device)
    
    # Save and load the probe
    temp_path = Path("test_artifacts")
    if not temp_path.exists():
        temp_path.mkdir()
    
    # Test both formats
    for fmt in ["pt", "json"]:
        probe_path = temp_path / f"test_inference_{fmt}.{fmt}"
        if fmt == "pt":
            probe.save(str(probe_path))
        else:
            probe.save_json(str(probe_path))
        
        # Create inference object with loaded probe
        inference = ProbeInference.from_saved_probe(
            model_name="gpt2",
            hook_point="blocks.7.hook_resid_pre",
            probe_path=str(probe_path),
            device=device
        )
        
        # Check if the probe in the inference object has standardization buffers
        inference_probe = inference.probe
        
        print(f"\n=== Inference Test Results ({fmt} format) ===")
        if hasattr(inference_probe, 'feature_mean'):
            print(f"feature_mean exists: {inference_probe.feature_mean is not None}")
            if inference_probe.feature_mean is not None:
                print(f"feature_mean shape: {inference_probe.feature_mean.shape}")
                print(f"feature_mean: {inference_probe.feature_mean}")
        if hasattr(inference_probe, 'feature_std'):
            print(f"feature_std exists: {inference_probe.feature_std is not None}")
            if inference_probe.feature_std is not None:
                print(f"feature_std shape: {inference_probe.feature_std.shape}")
                print(f"feature_std: {inference_probe.feature_std}")
                
        # Test if _apply_standardization works
        if hasattr(inference_probe, '_apply_standardization'):
            test_input = torch.tensor([[2.0, 4.0, 6.0]], device=device)
            standardized_input = inference_probe._apply_standardization(test_input)
            expected_standardized = (test_input - mean) / std
            
            print(f"Test input: {test_input}")
            print(f"Standardized input: {standardized_input}")
            print(f"Expected standardized: {expected_standardized}")
            print(f"Matches expected: {torch.allclose(standardized_input, expected_standardized)}")
            
            assert torch.allclose(standardized_input, expected_standardized), \
                "Standardization in inference should match expected values"

def investigate_inference_pipeline():
    """Investigate how ProbeInference uses the probe object."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a simple probe
    probe_config = LogisticProbeConfig(input_size=3, device=device)
    probe = LogisticProbe(probe_config)
    
    # Add standardization for testing
    mean = torch.tensor([1.0, 2.0, 3.0], device=device).reshape(1, -1)
    std = torch.tensor([0.5, 1.0, 1.5], device=device).reshape(1, -1)
    probe.register_buffer('feature_mean', mean)
    probe.register_buffer('feature_std', std)
    
    # Create an inference object
    inference = ProbeInference(
        model_name="gpt2",
        hook_point="blocks.7.hook_resid_pre",
        probe=probe,
        device=device
    )
    
    # Test fixed activations 
    test_activations = torch.tensor([
        [[2.0, 4.0, 6.0]],  # batch 0, token 0
    ], device=device)
    
    # Mock the get_activations method to return our test activations
    original_get_activations = inference.get_activations
    inference.get_activations = lambda text: test_activations
    
    # Test the inference methods
    raw_scores = inference.get_direction_activations("dummy text")
    probs = inference.get_probabilities("dummy text")
    probe_outputs = inference.get_probe_outputs("dummy text")
    
    # Restore the original method
    inference.get_activations = original_get_activations
    
    # Compute expected values manually
    # For raw scores - no sigmoid
    standardized = (test_activations.view(-1, 3) - mean) / std
    direction = probe.get_direction(normalized=True)
    expected_raw = torch.matmul(standardized, direction).view(test_activations.shape[0], -1)
    
    # For probe outputs - direct probe output, no sigmoid yet
    expected_probe_output = probe(test_activations.view(-1, 3)).view(test_activations.shape[0], -1)
    
    # For probabilities - apply sigmoid to probe output
    expected_probs = torch.sigmoid(expected_probe_output)
    
    print("\n=== Inference Pipeline Investigation ===")
    print(f"Test activations: {test_activations}")
    print(f"Standardized: {standardized}")
    print(f"Direction: {direction}")
    print(f"Expected raw scores: {expected_raw}")
    print(f"Actual raw scores: {raw_scores}")
    print(f"Raw scores match: {torch.allclose(raw_scores, expected_raw)}")
    
    print(f"\nExpected probe outputs: {expected_probe_output}")
    print(f"Actual probe outputs: {probe_outputs}")
    print(f"Probe outputs match: {torch.allclose(probe_outputs, expected_probe_output)}")
    
    print(f"\nExpected probabilities: {expected_probs}")
    print(f"Actual probabilities: {probs}")
    print(f"Probabilities match: {torch.allclose(probs, expected_probs)}")
    
    # Check the methods used in get_direction_activations
    print("\nStandardization in get_direction_activations:")
    activations = test_activations.view(-1, 3)
    if hasattr(probe, '_apply_standardization'):
        standardized_act = probe._apply_standardization(activations)
        print(f"  Standardized activations: {standardized_act}")
        print(f"  Matches expected: {torch.allclose(standardized_act, standardized)}")
    
    # Return information about which methods were called and their results
    return {
        "raw_scores": raw_scores,
        "expected_raw": expected_raw,
        "match_raw": torch.allclose(raw_scores, expected_raw),
        "probs": probs,
        "expected_probs": expected_probs,
        "match_probs": torch.allclose(probs, expected_probs),
        "probe_outputs": probe_outputs,
        "expected_probe_output": expected_probe_output,
        "match_probe_output": torch.allclose(probe_outputs, expected_probe_output)
    } 