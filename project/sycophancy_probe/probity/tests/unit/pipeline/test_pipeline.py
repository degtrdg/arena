import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import torch
import shutil
from typing import Type

from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from probity.collection.activation_store import ActivationStore
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes.linear_probe import BaseProbe, ProbeConfig
from probity.training.trainer import BaseProbeTrainer, BaseTrainerConfig


# --- Mock Classes ---


@pytest.fixture
def mock_probe_config():
    config = MagicMock()
    config.input_size = 10  # Example dimension
    return config


@pytest.fixture
def mock_trainer_config():
    config = MagicMock(spec=BaseTrainerConfig)
    config.device = "cpu"
    config.batch_size = 4
    return config


@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=TokenizedProbingDataset)
    dataset.examples = [Mock() for _ in range(10)]  # Mock examples
    dataset.tokenization_config = MagicMock(tokenizer_name="mock_tokenizer")
    dataset.position_types = {"POS1", "POS2"}
    dataset.__len__.return_value = 10
    return dataset


@pytest.fixture
def mock_probe_cls():
    cls = MagicMock(spec=Type[BaseProbe])
    instance = MagicMock(spec=BaseProbe)
    instance.to = MagicMock(return_value=instance)  # Simulate moving to device
    instance.load = MagicMock(return_value=instance)
    instance.load_json = MagicMock(return_value=instance)  # For ProbeVector loading
    cls.return_value = instance  # Mock constructor call
    cls.load = MagicMock(return_value=instance)  # Mock classmethod load
    cls.load_json = MagicMock(return_value=instance)  # Mock classmethod load_json
    return cls


@pytest.fixture
def mock_trainer_cls():
    mock_instance = MagicMock(spec=BaseProbeTrainer)

    mock_instance.prepare_supervised_data = MagicMock(
        return_value=(MagicMock(), MagicMock())
    )
    mock_instance.train = MagicMock(
        return_value={"train_loss": [0.1], "val_loss": [0.1]}
    )

    mock_cls = MagicMock(spec=Type[BaseProbeTrainer])
    mock_cls.return_value = mock_instance

    return mock_cls


@pytest.fixture
def mock_activation_store():
    store = MagicMock(spec=ActivationStore)
    mock_ds = MagicMock(spec=TokenizedProbingDataset)
    mock_ds.examples = [Mock()] * 10  # Mock examples list
    mock_ds.tokenization_config = MagicMock(tokenizer_name="mock_tokenizer")
    mock_ds.position_types = {"POS1", "POS2"}
    store.dataset = mock_ds  # Assign the more detailed dataset mock

    store.model_name = "mock_model"  # Add model_name attribute
    store.get_probe_data = MagicMock(
        return_value=(torch.randn(10, 10), torch.randint(0, 2, (10, 1)))
    )
    store.save = MagicMock()
    store.load = MagicMock(return_value=store)  # Class method returns instance
    ActivationStore.load = MagicMock(return_value=store)  # Mock classmethod load
    return store


@pytest.fixture
def base_pipeline_config(
    mock_dataset,
    mock_probe_cls,
    mock_probe_config,
    mock_trainer_cls,
    mock_trainer_config,
):
    return ProbePipelineConfig(
        dataset=mock_dataset,
        probe_cls=mock_probe_cls,
        probe_config=mock_probe_config,
        trainer_cls=mock_trainer_cls,
        trainer_config=mock_trainer_config,
        position_key="POS1",
        cache_dir=None,  # No cache by default
        model_name="mock_model",
        hook_points=["hp1"],
        activation_batch_size=4,
        device="cpu",
    )


@pytest.fixture
def pipeline_with_cache(base_pipeline_config, tmp_path):
    config = base_pipeline_config
    config.cache_dir = str(tmp_path / "test_cache")
    # Create the cache dir structure manually for the test
    # cache_base = Path(config.cache_dir)
    # cache_key = "mock_hash" # Assume a fixed hash for simplicity in setup
    # config_hash_path = cache_base / cache_key
    # config_hash_path.mkdir(parents=True, exist_ok=True)
    # (config_hash_path / "hp1").mkdir(exist_ok=True) # Store path for hookpoint
    return ProbePipeline(config)


@pytest.fixture
def pipeline_no_cache(base_pipeline_config):
    base_pipeline_config.cache_dir = None
    return ProbePipeline(base_pipeline_config)


# --- Test Cases ---


def test_pipeline_init_device_sync(
    mock_probe_config,
    mock_trainer_config,
    mock_dataset,
    mock_probe_cls,
    mock_trainer_cls,
):
    """Test device synchronization during initialization."""
    # Case 1: Trainer has device, pipeline adopts it
    trainer_config_cuda = MagicMock(spec=BaseTrainerConfig, device="cuda")
    probe_config_cpu = MagicMock(spec=ProbeConfig, device="cpu")
    config1 = ProbePipelineConfig(
        dataset=mock_dataset,
        probe_cls=mock_probe_cls,
        probe_config=probe_config_cpu,
        trainer_cls=mock_trainer_cls,
        trainer_config=trainer_config_cuda,
        position_key="POS1",
        model_name="m",
        hook_points=["hp1"],
        device="cpu",  # Pipeline device differs initially
    )
    pipeline1 = ProbePipeline(config1)
    assert pipeline1.config.device == "cuda"
    assert pipeline1.config.probe_config.device == "cuda"
    assert pipeline1.config.trainer_config.device == "cuda"

    # Case 2: Pipeline has device, trainer and probe adopt it
    trainer_config_nodev = MagicMock(spec=BaseTrainerConfig)
    # Remove device attribute if it exists to simulate it not being set
    if hasattr(trainer_config_nodev, "device"):
        del trainer_config_nodev.device
    # Use a basic mock but ensure 'device' exists for hasattr checks
    probe_config_nodev = MagicMock()
    probe_config_nodev.device = None  # Add placeholder for hasattr check
    if (
        hasattr(probe_config_nodev, "device") and probe_config_nodev.device is not None
    ):  # Defensive check
        del probe_config_nodev.device

    config2 = ProbePipelineConfig(
        dataset=mock_dataset,
        probe_cls=mock_probe_cls,
        probe_config=probe_config_nodev,
        trainer_cls=mock_trainer_cls,
        trainer_config=trainer_config_nodev,
        position_key="POS1",
        model_name="m",
        hook_points=["hp1"],
        device="mps",  # Pipeline sets device
    )
    pipeline2 = ProbePipeline(config2)
    assert pipeline2.config.device == "mps"
    assert pipeline2.config.probe_config.device == "mps"
    assert pipeline2.config.trainer_config.device == "mps"


@patch("probity.pipeline.pipeline.TransformerLensCollector")
def test_pipeline_run_no_cache(
    mock_collector_cls, pipeline_no_cache, mock_activation_store
):
    """Test pipeline run without cache."""
    # Setup mock collector
    mock_collector_instance = MagicMock()
    mock_collector_instance.collect.return_value = {"hp1": mock_activation_store}
    mock_collector_cls.return_value = mock_collector_instance

    # Mock trainer components
    mock_trainer_instance = pipeline_no_cache.config.trainer_cls.return_value

    # Run
    probe, history = pipeline_no_cache.run(hook_point="hp1")

    # Assertions
    mock_collector_instance.collect.assert_called_once_with(
        pipeline_no_cache.config.dataset
    )
    pipeline_no_cache.config.probe_cls.assert_called_once_with(
        pipeline_no_cache.config.probe_config
    )
    probe.to.assert_called_once_with("cpu")  # Check probe moved to device
    pipeline_no_cache.config.trainer_cls.assert_called_once_with(
        pipeline_no_cache.config.trainer_config
    )
    mock_trainer_instance.prepare_supervised_data.assert_called_once_with(
        mock_activation_store, pipeline_no_cache.config.position_key
    )
    mock_trainer_instance.train.assert_called_once()
    assert pipeline_no_cache.probe is probe
    assert history == {"train_loss": [0.1], "val_loss": [0.1]}


@patch("probity.pipeline.pipeline.Path")
@patch("probity.pipeline.pipeline.ActivationStore")
@patch("probity.pipeline.pipeline.TransformerLensCollector")
def test_pipeline_run_with_valid_cache(
    mock_collector_cls,
    mock_as_cls,
    mock_path_cls,
    pipeline_with_cache,
    mock_activation_store,
):
    """Test pipeline run with a valid cache found."""
    # --- Mock Cache Loading ---
    mock_base_path = MagicMock()
    mock_store_path = MagicMock()

    # Simulate Path(cache_dir) -> mock_base_path
    mock_path_cls.side_effect = lambda x: (
        mock_base_path if x == pipeline_with_cache.config.cache_dir else MagicMock()
    )

    # mock_base_path.exists() -> True
    mock_base_path.exists.return_value = True
    # mock_base_path / hook_point -> mock_store_path
    mock_base_path.__truediv__.return_value = mock_store_path
    # mock_store_path.exists() -> True
    mock_store_path.exists.return_value = True

    # mock_activation_store.load() -> mock_activation_store
    mock_as_cls.load.return_value = mock_activation_store

    # Make validation pass
    pipeline_with_cache._validate_cache_compatibility = MagicMock(return_value=True)
    # --- End Mock Cache Loading ---

    mock_collector_instance = MagicMock()  # Should not be called
    mock_collector_cls.return_value = mock_collector_instance
    mock_trainer_instance = pipeline_with_cache.config.trainer_cls.return_value

    # Run
    probe, history = pipeline_with_cache.run(hook_point="hp1")

    # Assertions
    mock_path_cls.assert_any_call(
        pipeline_with_cache.config.cache_dir
    )  # Check cache dir access
    mock_base_path.exists.assert_called_once()
    mock_base_path.__truediv__.assert_called_once_with(
        "hp1"
    )  # Check hook point subdir access
    mock_store_path.exists.assert_called_once()
    mock_as_cls.load.assert_called_once_with(str(mock_store_path))
    pipeline_with_cache._validate_cache_compatibility.assert_called_once_with(
        mock_activation_store, "hp1"
    )
    mock_collector_instance.collect.assert_not_called()  # Ensure collector was skipped
    pipeline_with_cache.config.probe_cls.assert_called_once()
    pipeline_with_cache.config.trainer_cls.assert_called_once()
    mock_trainer_instance.prepare_supervised_data.assert_called_once()
    mock_trainer_instance.train.assert_called_once()
    assert pipeline_with_cache.activation_stores["hp1"] is mock_activation_store


@patch("shutil.rmtree")
@patch("probity.pipeline.pipeline.Path")
@patch("probity.pipeline.pipeline.ActivationStore")
@patch("probity.pipeline.pipeline.TransformerLensCollector")
def test_pipeline_run_with_invalid_cache(
    mock_collector_cls,
    mock_as_cls,
    mock_path_cls,
    mock_rmtree,
    pipeline_with_cache,
    mock_activation_store,
):
    """Test pipeline run with an invalid cache (triggering collection)."""
    # --- Mock Cache Loading ---
    mock_base_path = MagicMock(spec=Path)
    mock_store_path = MagicMock(spec=Path)

    # Simulate Path(cache_dir) -> mock_base_path
    mock_path_cls.side_effect = lambda x: (
        mock_base_path
        if x == pipeline_with_cache.config.cache_dir
        else MagicMock(spec=Path)
    )

    mock_base_path.exists.return_value = True
    mock_base_path.__truediv__.return_value = mock_store_path
    mock_store_path.exists.return_value = True

    mock_as_cls.load.return_value = mock_activation_store

    # Make validation fail
    pipeline_with_cache._validate_cache_compatibility = MagicMock(return_value=False)
    # --- End Mock Cache Loading ---

    # Setup mock collector (will be called this time)
    collected_stores = {"hp1": mock_activation_store}
    mock_collector_instance = MagicMock()
    mock_collector_instance.collect.return_value = collected_stores
    mock_collector_cls.return_value = mock_collector_instance

    mock_trainer_instance = pipeline_with_cache.config.trainer_cls.return_value

    # Run
    probe, history = pipeline_with_cache.run(hook_point="hp1")

    # Assertions
    mock_path_cls.assert_any_call(pipeline_with_cache.config.cache_dir)
    mock_base_path.exists.assert_called_once()
    mock_base_path.__truediv__.assert_any_call("hp1")
    mock_store_path.exists.assert_called_once()
    mock_as_cls.load.assert_called_once_with(str(mock_store_path))
    pipeline_with_cache._validate_cache_compatibility.assert_called_once_with(
        mock_activation_store, "hp1"
    )
    mock_rmtree.assert_called_once_with(mock_base_path)  # Check cache cleared
    mock_collector_instance.collect.assert_called_once_with(
        pipeline_with_cache.config.dataset
    )  # Collector was called
    pipeline_with_cache.config.probe_cls.assert_called_once()
    pipeline_with_cache.config.trainer_cls.assert_called_once()
    mock_trainer_instance.prepare_supervised_data.assert_called_once()
    mock_trainer_instance.train.assert_called_once()
    assert pipeline_with_cache.activation_stores["hp1"] is collected_stores["hp1"]


@patch("probity.pipeline.pipeline.Path")
@patch("probity.pipeline.pipeline.ActivationStore")
@patch("probity.pipeline.pipeline.TransformerLensCollector")
def test_pipeline_run_cache_save(
    mock_collector_cls,
    mock_as_cls,
    mock_path_cls,
    pipeline_with_cache,
    mock_activation_store,
):
    """Test if activations are saved when cache_dir is set and collection happens."""
    # --- Mock Cache Loading (make it seem empty/invalid initially) ---
    mock_base_path = MagicMock(spec=Path)
    mock_store_path = MagicMock(spec=Path)
    mock_path_cls.side_effect = lambda x: (
        mock_base_path
        if x == pipeline_with_cache.config.cache_dir
        else MagicMock(spec=Path)
    )

    # Simulate cache dir DOES NOT exist initially to force collection & save
    mock_base_path.exists.return_value = False
    mock_base_path.mkdir = MagicMock()  # Mock mkdir call

    # --- End Mock Cache Loading ---

    # Setup mock collector
    collected_stores = {"hp1": mock_activation_store}
    mock_collector_instance = MagicMock()
    mock_collector_instance.collect.return_value = collected_stores
    mock_collector_cls.return_value = mock_collector_instance

    # mock_trainer_instance = pipeline_with_cache.config.trainer_cls.return_value # Unused

    # Need to mock the __truediv__ call that happens during the save loop
    mock_base_path.__truediv__.return_value = mock_store_path

    # Run
    probe, history = pipeline_with_cache.run(hook_point="hp1")

    # Assertions
    mock_path_cls.assert_any_call(pipeline_with_cache.config.cache_dir)
    mock_base_path.exists.assert_called_once()  # Check existence
    # Store path existence shouldn't be checked initially
    # mock_store_path.exists.assert_not_called()
    # Check base dir creation
    mock_base_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_collector_instance.collect.assert_called_once()
    # Check __truediv__ was called to construct the save path
    mock_base_path.__truediv__.assert_called_with("hp1")
    # Check store was saved
    mock_activation_store.save.assert_called_once_with(str(mock_store_path))


@patch(
    "probity.pipeline.pipeline.ProbePipelineConfig"
)  # We need to mock the config used internally
@patch("probity.pipeline.pipeline.torch")
@patch("probity.pipeline.pipeline.Path")
def test_pipeline_load(
    mock_path_cls,
    mock_torch,
    MockPipelineConfig,
    mock_probe_cls,
    mock_trainer_cls,
    tmp_path,
):
    """Test loading a pipeline from a saved state."""
    load_dir = tmp_path / "saved_pipeline"
    load_dir.mkdir()
    probe_path = load_dir / "probe.pt"
    config_path = load_dir / "config.pt"

    # Mock Path objects
    mock_load_path = MagicMock(spec=Path)
    mock_probe_path = MagicMock(spec=Path)
    mock_config_path = MagicMock(spec=Path)
    mock_vector_path = MagicMock(spec=Path)  # For probe_vector.json

    mock_path_cls.side_effect = lambda x: {
        str(load_dir): mock_load_path,
        str(probe_path): mock_probe_path,
        str(config_path): mock_config_path,
        str(load_dir / "probe_vector.json"): mock_vector_path,
    }.get(
        str(x), MagicMock(spec=Path)
    )  # Return specific mocks or generic one

    mock_load_path.__truediv__.side_effect = lambda x: {
        "config.pt": mock_config_path,
        "probe.pt": mock_probe_path,
        "probe_vector.json": mock_vector_path,
    }.get(x)

    # Mock loaded config
    mock_loaded_config = MagicMock(spec=ProbePipelineConfig)
    mock_loaded_config.probe_cls = mock_probe_cls  # The class itself
    mock_loaded_config.trainer_config = MagicMock(
        device="cuda"
    )  # Config object has device
    mock_loaded_config.probe_config = MagicMock()  # Needs to exist
    mock_torch.load.return_value = mock_loaded_config
    MockPipelineConfig.return_value = (
        mock_loaded_config  # Ensure constructor uses mocked config
    )

    # Mock probe loading
    mock_probe_instance = (
        mock_probe_cls.return_value
    )  # Get the instance created by the fixture
    # Simulate only probe.pt exists
    mock_probe_path.exists.return_value = True
    mock_vector_path.exists.return_value = False

    # Call the class method
    pipeline = ProbePipeline.load(str(load_dir))

    # Assertions
    mock_path_cls.assert_any_call(str(load_dir))
    mock_load_path.__truediv__.assert_any_call("config.pt")
    mock_torch.load.assert_called_once_with(str(mock_config_path))
    # Ensure probe_cls (which is a mock) was called to create the probe instance
    # It's called inside the load method, not necessarily inside the constructor
    mock_probe_cls.load.assert_called_once_with(str(mock_probe_path))
    mock_probe_instance.to.assert_called_once_with(
        "cuda"
    )  # Loaded probe moved to device from config
    assert pipeline.probe is mock_probe_instance
    assert pipeline.config is mock_loaded_config


def test_validate_cache_compatibility(
    pipeline_no_cache, mock_activation_store, mock_dataset
):
    """Test cache validation logic."""
    pipeline = pipeline_no_cache  # Use a pipeline instance for access to config

    # Case 1: Compatible
    store_compatible = MagicMock(spec=ActivationStore)
    comp_ds = MagicMock(spec=TokenizedProbingDataset)
    comp_ds.examples = [Mock()] * 10
    comp_ds.tokenization_config = MagicMock(tokenizer_name="mock_tokenizer")
    comp_ds.position_types = {"POS1", "POS2"}
    store_compatible.dataset = comp_ds
    store_compatible.model_name = "mock_model"  # Match model name

    assert pipeline._validate_cache_compatibility(store_compatible, "hp1") is True

    # Case 2: Dataset size mismatch
    store_diff_size = MagicMock(spec=ActivationStore)
    diff_size_ds = MagicMock(spec=TokenizedProbingDataset)
    diff_size_ds.examples = [Mock()] * 5  # Different size
    diff_size_ds.tokenization_config = MagicMock(tokenizer_name="mock_tokenizer")
    diff_size_ds.position_types = {"POS1", "POS2"}
    store_diff_size.dataset = diff_size_ds
    store_diff_size.model_name = "mock_model"
    assert pipeline._validate_cache_compatibility(store_diff_size, "hp1") is False

    # Case 3: Tokenizer mismatch
    store_diff_tokenizer = MagicMock(spec=ActivationStore)
    diff_tok_ds = MagicMock(spec=TokenizedProbingDataset)
    diff_tok_ds.examples = [Mock()] * 10
    diff_tok_ds.tokenization_config = MagicMock(
        tokenizer_name="other_tokenizer"
    )  # Diff tokenizer
    diff_tok_ds.position_types = {"POS1", "POS2"}
    store_diff_tokenizer.dataset = diff_tok_ds
    store_diff_tokenizer.model_name = "mock_model"
    assert pipeline._validate_cache_compatibility(store_diff_tokenizer, "hp1") is False

    # Case 4: Position types mismatch
    store_diff_pos = MagicMock(spec=ActivationStore)
    diff_pos_ds = MagicMock(spec=TokenizedProbingDataset)
    diff_pos_ds.examples = [Mock()] * 10
    diff_pos_ds.tokenization_config = MagicMock(tokenizer_name="mock_tokenizer")
    diff_pos_ds.position_types = {"POS_X"}  # Different position types
    store_diff_pos.dataset = diff_pos_ds
    store_diff_pos.model_name = "mock_model"
    assert pipeline._validate_cache_compatibility(store_diff_pos, "hp1") is False

    # Case 5: Model name mismatch
    store_diff_model = MagicMock(spec=ActivationStore)
    diff_model_ds = MagicMock(spec=TokenizedProbingDataset)
    diff_model_ds.examples = [Mock()] * 10
    diff_model_ds.tokenization_config = MagicMock(tokenizer_name="mock_tokenizer")
    diff_model_ds.position_types = {"POS1", "POS2"}
    store_diff_model.dataset = diff_model_ds
    store_diff_model.model_name = "other_model"  # Different model name
    assert pipeline._validate_cache_compatibility(store_diff_model, "hp1") is False


@patch("hashlib.md5")
@patch("json.dumps")
def test_get_cache_key(mock_json_dumps, mock_md5, pipeline_no_cache):
    """Test cache key generation."""
    # Mock hashlib and json.dumps
    mock_hash_obj = MagicMock()
    mock_hash_obj.hexdigest.return_value = "mock_hash_value"
    mock_md5.return_value = mock_hash_obj
    mock_json_dumps.return_value = '{"json_string": true}'

    # Call the method
    key = pipeline_no_cache._get_cache_key()

    # Assertions
    expected_dict = {
        "model_name": "mock_model",
        "hook_points": ["hp1"],
        "position_key": "POS1",
        "tokenizer_name": "mock_tokenizer",
        "dataset_size": 10,
        "position_types": sorted(list({"POS1", "POS2"})),  # Ensure order consistency
    }
    # Check json.dumps called with correct dict structure and sort_keys=True
    mock_json_dumps.assert_called_once()
    call_args, call_kwargs = mock_json_dumps.call_args
    assert (
        call_args[0] == expected_dict
    )  # This should now pass due to sorted list in source
    assert call_kwargs.get("sort_keys") is True

    # Check hashlib.md5 was called with the encoded json string
    mock_md5.assert_called_once_with('{"json_string": true}'.encode())
    mock_hash_obj.hexdigest.assert_called_once()
    assert key == "mock_hash_value"


@patch("probity.pipeline.pipeline.Path")
def test_get_cache_path(mock_path_cls, pipeline_with_cache):
    """Test cache path generation."""
    # Mock Path and _get_cache_key
    mock_base_path = MagicMock(spec=Path)
    mock_final_path = MagicMock(spec=Path)
    mock_path_cls.return_value = mock_base_path
    mock_base_path.__truediv__.return_value = mock_final_path
    pipeline_with_cache._get_cache_key = MagicMock(return_value="mock_config_hash")

    # Call the method
    cache_path = pipeline_with_cache._get_cache_path()

    # Assertions
    mock_path_cls.assert_called_once_with(pipeline_with_cache.config.cache_dir)
    pipeline_with_cache._get_cache_key.assert_called_once()
    mock_base_path.__truediv__.assert_called_once_with("mock_config_hash")
    assert cache_path is mock_final_path


def test_run_without_hook_point_specified(pipeline_no_cache):
    """Test run uses the first hook point if none is specified."""
    pipeline_no_cache.config.hook_points = ["hp_first", "hp_second"]

    # Mock activation stores loading
    mock_store1 = MagicMock(spec=ActivationStore)
    mock_store2 = MagicMock(spec=ActivationStore)
    pipeline_no_cache._load_or_collect_activations = MagicMock(
        return_value={"hp_first": mock_store1, "hp_second": mock_store2}
    )

    mock_trainer_instance = pipeline_no_cache.config.trainer_cls.return_value

    # Run without specifying hook_point
    pipeline_no_cache.run()

    # Assert that prepare_supervised_data was called with the first store
    mock_trainer_instance.prepare_supervised_data.assert_called_once_with(
        mock_store1, pipeline_no_cache.config.position_key
    )


def test_run_with_invalid_hook_point(pipeline_no_cache):
    """Test run raises ValueError for an unknown hook point."""
    pipeline_no_cache.config.hook_points = ["hp_known"]
    # Mock activation stores loading
    mock_store = MagicMock(spec=ActivationStore)
    pipeline_no_cache._load_or_collect_activations = MagicMock(
        return_value={"hp_known": mock_store}
    )

    with pytest.raises(ValueError, match="Hook point hp_unknown not found"):
        pipeline_no_cache.run(hook_point="hp_unknown")


# Clean up cache directory if it exists (safety measure)
@pytest.fixture(autouse=True)
def cleanup_cache(tmp_path):
    cache_dir = tmp_path / "test_cache"
    yield
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
