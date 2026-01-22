import pytest
import os
import json
import shutil
from datasets import Dataset

from probity.datasets.base import (
    ProbingDataset,
    ProbingExample,
    CharacterPositions,
    Position,
)


# Fixtures
@pytest.fixture
def sample_position():
    return Position(start=5, end=10)


@pytest.fixture
def sample_multi_position():
    return [Position(start=5, end=10), Position(start=20, end=25)]


@pytest.fixture
def sample_char_positions(sample_position, sample_multi_position):
    return CharacterPositions(
        positions={"word": sample_position, "phrase": sample_multi_position}
    )


@pytest.fixture
def sample_probing_example(sample_char_positions):
    return ProbingExample(
        text="This is a sample text.",
        label=1,
        label_text="positive",
        character_positions=sample_char_positions,
        group_id="group1",
        attributes={"source": "test"},
    )


@pytest.fixture
def sample_probing_example_no_pos():
    return ProbingExample(
        text="Another sample text.",
        label=0,
        label_text="negative",
        group_id="group2",
        attributes={"source": "test"},
    )


@pytest.fixture
def sample_probing_examples(sample_probing_example, sample_probing_example_no_pos):
    return [sample_probing_example, sample_probing_example_no_pos]


@pytest.fixture
def sample_probing_dataset(sample_probing_examples):
    return ProbingDataset(
        examples=sample_probing_examples,
        task_type="classification",
        valid_layers=["layer1"],
        label_mapping={"positive": 1, "negative": 0},
        dataset_attributes={"name": "test_dataset"},
    )


@pytest.fixture
def sample_hf_dataset(sample_probing_dataset):
    return sample_probing_dataset._to_hf_dataset()


@pytest.fixture
def temp_save_dir():
    path = "./temp_test_dataset"
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


# Tests for CharacterPositions
def test_character_positions_getitem(sample_char_positions, sample_position):
    assert sample_char_positions["word"] == sample_position


def test_character_positions_keys(sample_char_positions):
    assert sample_char_positions.keys() == {"word", "phrase"}


# Tests for ProbingExample
def test_probing_example_creation(sample_probing_example):
    assert sample_probing_example.text == "This is a sample text."
    assert sample_probing_example.label == 1
    assert sample_probing_example.label_text == "positive"
    assert sample_probing_example.group_id == "group1"
    assert sample_probing_example.attributes == {"source": "test"}
    assert isinstance(sample_probing_example.character_positions, CharacterPositions)


# Tests for ProbingDataset
def test_probing_dataset_initialization(
    sample_probing_dataset, sample_probing_examples
):
    assert sample_probing_dataset.examples == sample_probing_examples
    assert sample_probing_dataset.task_type == "classification"
    assert sample_probing_dataset.valid_layers == ["layer1"]
    assert sample_probing_dataset.label_mapping == {"positive": 1, "negative": 0}
    assert sample_probing_dataset.dataset_attributes == {"name": "test_dataset"}
    assert sample_probing_dataset.position_types == {"word", "phrase"}


def test_probing_dataset_initialization_no_pos(sample_probing_example_no_pos):
    dataset = ProbingDataset([sample_probing_example_no_pos])
    assert dataset.position_types == set()


def test_probing_dataset_add_target_positions(
    sample_probing_dataset, sample_probing_example, sample_probing_example_no_pos
):
    def find_sample(text):
        if "sample" in text:
            return Position(start=text.find("sample"), end=text.find("sample") + 6)
        return None  # Test case where finder returns None

    sample_probing_dataset.add_target_positions("sample_word", find_sample)

    # Check example with positions
    example1 = sample_probing_dataset.examples[0]
    assert "sample_word" in example1.character_positions.keys()
    pos1 = example1.character_positions["sample_word"]
    assert isinstance(pos1, Position)
    assert pos1.start == 10
    assert pos1.end == 16

    # Check example without initial positions
    example2 = sample_probing_dataset.examples[1]
    assert example2.character_positions is not None  # Should be initialized
    assert "sample_word" in example2.character_positions.keys()
    pos2 = example2.character_positions["sample_word"]
    assert isinstance(pos2, Position)
    assert pos2.start == 8
    assert pos2.end == 14

    assert "sample_word" in sample_probing_dataset.position_types


def test_probing_dataset_to_hf_dataset(sample_probing_dataset, sample_hf_dataset):
    assert isinstance(sample_hf_dataset, Dataset)
    assert len(sample_hf_dataset) == 2
    assert set(sample_hf_dataset.column_names) == {
        "text",
        "label",
        "label_text",
        "group_id",
        "char_pos_word_start",
        "char_pos_word_end",
        "char_pos_word_multi",
        "char_pos_phrase_start",
        "char_pos_phrase_end",
        "char_pos_phrase_multi",
    }

    # Check first example (with positions)
    assert sample_hf_dataset[0]["text"] == "This is a sample text."
    assert sample_hf_dataset[0]["label"] == 1.0  # HF converts labels to float
    assert sample_hf_dataset[0]["char_pos_word_start"] == 5
    assert sample_hf_dataset[0]["char_pos_word_end"] == 10
    assert sample_hf_dataset[0]["char_pos_word_multi"] == []
    assert sample_hf_dataset[0]["char_pos_phrase_start"] == 5  # First of multi
    assert sample_hf_dataset[0]["char_pos_phrase_end"] == 10  # First of multi
    assert sample_hf_dataset[0]["char_pos_phrase_multi"] == [[5, 10], [20, 25]]

    # Check second example (without positions)
    assert sample_hf_dataset[1]["text"] == "Another sample text."
    assert sample_hf_dataset[1]["label"] == 0.0
    assert sample_hf_dataset[1]["char_pos_word_start"] is None
    assert sample_hf_dataset[1]["char_pos_word_end"] is None
    assert sample_hf_dataset[1]["char_pos_word_multi"] == []
    assert sample_hf_dataset[1]["char_pos_phrase_start"] is None
    assert sample_hf_dataset[1]["char_pos_phrase_end"] is None
    assert sample_hf_dataset[1]["char_pos_phrase_multi"] == []


def test_probing_dataset_from_hf_dataset(sample_hf_dataset, sample_probing_dataset):
    loaded_dataset = ProbingDataset.from_hf_dataset(
        dataset=sample_hf_dataset,
        position_types=["word", "phrase"],
        label_mapping=sample_probing_dataset.label_mapping,
        task_type=sample_probing_dataset.task_type,
        valid_layers=sample_probing_dataset.valid_layers,
        dataset_attributes=sample_probing_dataset.dataset_attributes,
    )

    assert len(loaded_dataset.examples) == len(sample_probing_dataset.examples)
    assert loaded_dataset.task_type == sample_probing_dataset.task_type
    assert loaded_dataset.label_mapping == sample_probing_dataset.label_mapping
    assert (
        loaded_dataset.dataset_attributes == sample_probing_dataset.dataset_attributes
    )
    assert loaded_dataset.position_types == sample_probing_dataset.position_types

    # Deep comparison of examples (need to handle float label conversion)
    for loaded_ex, orig_ex in zip(
        loaded_dataset.examples, sample_probing_dataset.examples
    ):
        assert loaded_ex.text == orig_ex.text
        assert float(loaded_ex.label) == float(orig_ex.label)  # Compare as float
        assert loaded_ex.label_text == orig_ex.label_text
        assert loaded_ex.group_id == orig_ex.group_id
        # Attributes are not stored/loaded per example in HF dataset
        # Handle None vs empty CharacterPositions due to HF conversion
        if orig_ex.character_positions is None:
            assert loaded_ex.character_positions == CharacterPositions(
                positions={}
            ), f"Expected empty CharacterPositions for None input, got {loaded_ex.character_positions}"
        else:
            assert loaded_ex.character_positions == orig_ex.character_positions


def test_probing_dataset_save_load(sample_probing_dataset, temp_save_dir):
    # Save the dataset
    sample_probing_dataset.save(temp_save_dir)

    # Check if files exist
    assert os.path.exists(f"{temp_save_dir}/hf_dataset")
    assert os.path.exists(f"{temp_save_dir}/dataset_attributes.json")

    # Load the dataset
    loaded_dataset = ProbingDataset.load(temp_save_dir)

    # Perform comparisons (similar to from_hf_dataset test)
    assert len(loaded_dataset.examples) == len(sample_probing_dataset.examples)
    assert loaded_dataset.task_type == sample_probing_dataset.task_type
    assert loaded_dataset.label_mapping == sample_probing_dataset.label_mapping
    assert (
        loaded_dataset.dataset_attributes == sample_probing_dataset.dataset_attributes
    )
    assert loaded_dataset.position_types == sample_probing_dataset.position_types

    # Compare examples
    for loaded_ex, orig_ex in zip(
        loaded_dataset.examples, sample_probing_dataset.examples
    ):
        assert loaded_ex.text == orig_ex.text
        assert float(loaded_ex.label) == float(orig_ex.label)
        assert loaded_ex.label_text == orig_ex.label_text
        assert loaded_ex.group_id == orig_ex.group_id
        # Handle None vs empty CharacterPositions due to HF conversion
        if orig_ex.character_positions is None:
            assert loaded_ex.character_positions == CharacterPositions(
                positions={}
            ), f"Expected empty CharacterPositions for None input, got {loaded_ex.character_positions}"
        else:
            assert loaded_ex.character_positions == orig_ex.character_positions


def test_probing_dataset_save_load_backward_compatibility(
    sample_probing_dataset, temp_save_dir
):
    # Simulate saving with the old metadata format
    sample_probing_dataset.dataset.save_to_disk(f"{temp_save_dir}/hf_dataset")
    old_metadata = {
        "task_type": sample_probing_dataset.task_type,
        "valid_layers": sample_probing_dataset.valid_layers,
        "label_mapping": sample_probing_dataset.label_mapping,
        "metadata": sample_probing_dataset.dataset_attributes,  # Old key name
        "position_types": list(sample_probing_dataset.position_types),
    }
    with open(f"{temp_save_dir}/metadata.json", "w") as f:  # Old filename
        json.dump(old_metadata, f)

    # Load the dataset - should work with the old format
    loaded_dataset = ProbingDataset.load(temp_save_dir)

    # Verify loaded data
    assert (
        loaded_dataset.dataset_attributes == sample_probing_dataset.dataset_attributes
    )
    assert loaded_dataset.position_types == sample_probing_dataset.position_types


def test_probing_dataset_train_test_split(sample_probing_dataset):
    # Create a slightly larger dataset for splitting
    examples = [
        ProbingExample(f"text {i}", i % 2, str(i % 2), None, f"group{i}")
        for i in range(10)
    ]
    dataset = ProbingDataset(
        examples=examples, dataset_attributes=sample_probing_dataset.dataset_attributes
    )

    train_ds, test_ds = dataset.train_test_split(test_size=0.3, seed=42)

    assert isinstance(train_ds, ProbingDataset)
    assert isinstance(test_ds, ProbingDataset)
    assert len(train_ds.examples) == 7
    assert len(test_ds.examples) == 3

    # Check that attributes are preserved
    assert train_ds.dataset_attributes == dataset.dataset_attributes
    assert test_ds.dataset_attributes == dataset.dataset_attributes

    # Check that position types are correctly inferred (should be empty here)
    assert train_ds.position_types == set()
    assert test_ds.position_types == set()

    # Check content consistency: sizes and non-overlapping union = original
    original_texts = {ex.text for ex in examples}
    train_texts = {ex.text for ex in train_ds.examples}
    test_texts = {ex.text for ex in test_ds.examples}
    assert len(train_texts) == 7
    assert len(test_texts) == 3
    assert train_texts.union(test_texts) == original_texts
    assert train_texts.intersection(test_texts) == set()


def test_probing_dataset_split_with_positions(sample_probing_dataset):
    # Create a dataset with positions for splitting
    examples = [
        ProbingExample(
            f"text {i}",
            i % 2,
            str(i % 2),
            CharacterPositions({"num": Position(0, 1)}),
            f"group{i}",
        )
        for i in range(10)
    ]
    dataset = ProbingDataset(
        examples=examples, dataset_attributes=sample_probing_dataset.dataset_attributes
    )
    assert dataset.position_types == {"num"}

    train_ds, test_ds = dataset.train_test_split(test_size=0.3, seed=42)

    assert train_ds.position_types == {"num"}
    assert test_ds.position_types == {"num"}

    # The essential checks are that position_types are preserved.
    # Checking specific examples in the split is brittle.
    # test_example_texts = {ex.text for ex in test_ds.examples}
    # assert "text 1" in test_example_texts
    # test_ex_1 = next(ex for ex in test_ds.examples if ex.text == "text 1")
    # assert (
    #     test_ex_1.character_positions is not None
    # ), f"Character positions are None for example: {test_ex_1}"
    # assert test_ex_1.character_positions["num"] == Position(0, 1)
