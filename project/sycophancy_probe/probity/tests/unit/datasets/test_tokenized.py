import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions
from probity.datasets.position_finder import Position, PositionFinder
from probity.datasets.tokenized import (
    TokenizedProbingDataset,
    TokenizedProbingExample,
    TokenizationConfig,
    TokenPositions,
)
import dataclasses
import tempfile
import os
import shutil
import datasets  # Added import for datasets.Dataset
from typing import cast


# Fixtures
@pytest.fixture(
    scope="module"
)  # Use module scope for tokenizer fixtures to load only once
def tokenizer_right_pad() -> PreTrainedTokenizerFast:
    """Fixture for a right-padding gpt2 tokenizer."""
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("gpt2", use_fast=True),
    )
    tokenizer.padding_side = "right"
    # GPT2 uses EOS token for padding if pad_token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def tokenizer_left_pad() -> PreTrainedTokenizerFast:
    """Fixture for a left-padding gpt2 tokenizer."""
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("gpt2", use_fast=True),
    )
    # Important: Must explicitly set padding_side on instance
    tokenizer.padding_side = "left"
    # GPT2 uses EOS token for padding if pad_token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def simple_probing_dataset() -> ProbingDataset:
    """Fixture for a simple ProbingDataset."""
    examples = [
        ProbingExample(
            text="This is the first example.",  # len=27
            label=0,
            label_text="class_0",
            character_positions=CharacterPositions(
                {"FIRST_WORD": Position(0, 4)}
            ),  # "This"
            attributes={"id": "ex1"},
        ),
        ProbingExample(
            text="A second, longer example here.",  # len=30
            label=1,
            label_text="class_1",
            character_positions=CharacterPositions(
                {"FIRST_WORD": Position(0, 1)}
            ),  # "A"
            attributes={"id": "ex2"},
        ),
        ProbingExample(
            text="Third.",  # len=6
            label=0,
            label_text="class_0",
            # No character positions
            attributes={"id": "ex3"},
        ),
    ]
    return ProbingDataset(
        examples=examples,
        task_type="classification",
        label_mapping={"class_0": 0, "class_1": 1},
        dataset_attributes={"name": "simple_test_set"},
    )


@pytest.fixture
def dataset_with_list_pos() -> ProbingDataset:
    """Fixture for a ProbingDataset with list positions."""
    examples = [
        ProbingExample(
            text="Multiple targets here and here.",  # len=31
            label=1,
            label_text="multi",
            character_positions=CharacterPositions(
                {"TARGET": [Position(9, 16), Position(26, 30)]}  # "targets", "here"
            ),
        ),
    ]
    return ProbingDataset(
        examples=examples, task_type="classification", label_mapping={"multi": 1}
    )


# Helper to get padding length
def get_pad_length(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    add_special_tokens: bool,
) -> int:
    """Calculates padding length for a given text."""
    tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return max(0, max_length - len(tokens))


# Test Cases
def test_initialization():
    config = TokenizationConfig(
        tokenizer_name="test-tokenizer",
        tokenizer_kwargs={},
        vocab_size=100,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        padding_side="right",
    )
    # Ensure CharacterPositions takes dict[str, Union[Position, List[Position]]]
    char_pos = CharacterPositions({"test": Position(0, 1)})
    examples = [
        TokenizedProbingExample(
            text="test",
            tokens=[10, 20],
            label=0,
            label_text="test_label",
            character_positions=char_pos,
        )
    ]
    dataset = TokenizedProbingDataset(
        examples=examples, tokenization_config=config, task_type="classification"
    )
    assert len(dataset) == 1
    assert isinstance(dataset.examples[0], TokenizedProbingExample)
    assert dataset.tokenization_config == config


def test_initialization_validation_fail():
    config = TokenizationConfig(
        tokenizer_name="test-tokenizer",
        tokenizer_kwargs={},
        vocab_size=100,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        padding_side="right",
    )
    # Example with invalid token ID (150 >= vocab_size 100)
    examples = [
        TokenizedProbingExample(
            text="test", tokens=[10, 150], label=0, label_text="fail_label"
        )
    ]
    with pytest.raises(ValueError, match="Invalid token ID found"):
        TokenizedProbingDataset(
            examples=examples, tokenization_config=config, task_type="classification"
        )


def test_from_probing_dataset_right_pad(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted max_length for GPT2 tokenization if needed
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,  # Ensure tensors aren't returned by tokenizer directly
    )

    assert len(tokenized_dataset) == 3
    assert isinstance(tokenized_dataset, TokenizedProbingDataset)
    assert tokenized_dataset.position_types == {"FIRST_WORD"}
    assert (
        tokenized_dataset.tokenization_config.tokenizer_name
        == tokenizer_right_pad.name_or_path
    )
    assert tokenized_dataset.tokenization_config.padding_side == "right"
    assert (
        tokenized_dataset.tokenization_config.pad_token_id
        == tokenizer_right_pad.pad_token_id
    )
    assert "max_length" in tokenized_dataset.tokenization_config.tokenizer_kwargs
    assert (
        tokenized_dataset.tokenization_config.tokenizer_kwargs["max_length"]
        == max_length
    )
    assert (
        tokenized_dataset.tokenization_config.tokenizer_kwargs["add_special_tokens"]
        is add_special
    )

    # Check example 1: "This is the first example."
    ex1 = cast(TokenizedProbingExample, tokenized_dataset.examples[0])
    assert isinstance(ex1, TokenizedProbingExample)
    assert len(ex1.tokens) == max_length
    assert ex1.attention_mask is not None
    assert len(ex1.attention_mask) == max_length
    assert ex1.attention_mask[-1] == 0  # Check padding
    # GPT2 adds BOS token when add_special_tokens=True - Check if this holds
    # assert ex1.tokens[0] == tokenizer_right_pad.bos_token_id # Commenting out BOS check
    assert ex1.token_positions is not None
    assert "FIRST_WORD" in ex1.token_positions.keys()
    # Character pos (0, 4) -> "This" -> token position 0 (assuming no BOS)
    assert ex1.token_positions["FIRST_WORD"] == 0

    # Check example 2: "A second, longer example here."
    ex2 = cast(TokenizedProbingExample, tokenized_dataset.examples[1])
    assert len(ex2.tokens) == max_length
    assert ex2.attention_mask is not None
    assert ex2.token_positions is not None
    assert "FIRST_WORD" in ex2.token_positions.keys()
    # Character pos (0, 1) -> " A" -> token position 0 (assuming no BOS)
    assert ex2.token_positions["FIRST_WORD"] == 0

    # Check example 3: "Third." (no character positions)
    ex3 = cast(TokenizedProbingExample, tokenized_dataset.examples[2])
    assert len(ex3.tokens) == max_length
    assert ex3.attention_mask is not None
    # If positions were defined in the original dataset, TokenPositions should be created
    # If NO positions were defined for this example, token_positions should be None
    assert ex3.token_positions is None


def test_from_probing_dataset_left_pad(simple_probing_dataset, tokenizer_left_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_left_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    assert len(tokenized_dataset) == 3
    assert tokenized_dataset.tokenization_config.padding_side == "left"
    assert (
        tokenized_dataset.tokenization_config.pad_token_id
        == tokenizer_left_pad.pad_token_id
    )

    # Check example 1: "This is the first example."
    ex1 = cast(TokenizedProbingExample, tokenized_dataset.examples[0])
    assert len(ex1.tokens) == max_length
    assert ex1.attention_mask is not None
    assert ex1.tokens[0] == tokenizer_left_pad.pad_token_id  # Check padding
    # Find first non-pad token
    first_real_token_index = next(
        (i for i, t in enumerate(ex1.tokens) if t != tokenizer_left_pad.pad_token_id),
        -1,
    )
    assert first_real_token_index != -1
    # First real token should be BOS for GPT2 - Check if this holds
    # assert ex1.tokens[first_real_token_index] == tokenizer_left_pad.bos_token_id # Commenting out BOS check
    assert ex1.attention_mask[0] == 0  # Check attention mask padding
    assert ex1.token_positions is not None
    assert "FIRST_WORD" in ex1.token_positions.keys()
    # "This" is token 1 in the unpadded sequence [BOS] This is the first example . [EOS]
    pad_len = get_pad_length(ex1.text, tokenizer_left_pad, max_length, add_special)

    # Dynamically calculate expected position based on whether BOS was actually added
    unpadded_pos = 0  # Assuming position 0 if no BOS
    unpadded_tokens_no_special = tokenizer_left_pad.encode(
        ex1.text, add_special_tokens=False
    )
    unpadded_tokens_with_special = tokenizer_left_pad.encode(
        ex1.text, add_special_tokens=True
    )
    if (
        len(unpadded_tokens_with_special) > len(unpadded_tokens_no_special)
        and unpadded_tokens_with_special[0] == tokenizer_left_pad.bos_token_id
    ):
        unpadded_pos = 1  # Position is 1 if BOS was added

    # Expected position = unpadded_pos + pad_len
    expected_pos = unpadded_pos + pad_len
    assert ex1.token_positions["FIRST_WORD"] == expected_pos

    # Check example 3: "Third." (shortest)
    ex3 = cast(TokenizedProbingExample, tokenized_dataset.examples[2])
    assert len(ex3.tokens) == max_length
    assert ex3.attention_mask is not None
    assert ex3.tokens[0] == tokenizer_left_pad.pad_token_id
    assert ex3.token_positions is None  # No positions defined


def test_from_probing_dataset_list_pos(dataset_with_list_pos, tokenizer_right_pad):
    max_length = 20
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=dataset_with_list_pos,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )
    assert len(tokenized_dataset) == 1
    ex = cast(TokenizedProbingExample, tokenized_dataset.examples[0])
    assert ex.token_positions is not None
    assert "TARGET" in ex.token_positions.keys()
    assert isinstance(ex.token_positions["TARGET"], list)
    # Text: "Multiple targets here and here."
    # GPT2 Tokens (approx): [BOS] Multiple targets here and here . [EOS] [PAD] ...
    # "targets" -> char (9, 16) -> might be multiple tokens, need dynamic check
    # "here" -> char (26, 30) -> might be one token

    # Dynamic check for positions
    pos_targets = PositionFinder.convert_to_token_position(
        Position(9, 16), ex.text, tokenizer_right_pad, add_special_tokens=add_special
    )
    pos_here = PositionFinder.convert_to_token_position(
        Position(26, 30), ex.text, tokenizer_right_pad, add_special_tokens=add_special
    )
    # Assuming convert_to_token_position returns the first token index if span covers multiple
    expected_positions = [pos_targets, pos_here]

    assert ex.token_positions["TARGET"] == expected_positions


def test_get_batch_tensors_right_pad(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    batch = tokenized_dataset.get_batch_tensors(indices=[0, 1], pad=True)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "positions" in batch
    # Cast to Tensor for shape check
    input_ids_tensor = cast(torch.Tensor, batch["input_ids"])
    attention_mask_tensor = cast(torch.Tensor, batch["attention_mask"])
    assert isinstance(input_ids_tensor, torch.Tensor)
    assert isinstance(attention_mask_tensor, torch.Tensor)
    assert isinstance(batch["positions"], dict)

    # Shape check: (batch_size, sequence_length)
    assert input_ids_tensor.shape == (2, max_length)
    assert attention_mask_tensor.shape == (2, max_length)

    # Position check
    assert "FIRST_WORD" in batch["positions"]
    assert isinstance(batch["positions"]["FIRST_WORD"], torch.Tensor)
    # Expected positions [0, 0] for "This" and " A" (token index 0)
    assert torch.equal(batch["positions"]["FIRST_WORD"], torch.tensor([0, 0]))

    # Padding check (right padding)
    assert input_ids_tensor[0, -1] == tokenizer_right_pad.pad_token_id
    assert attention_mask_tensor[0, -1] == 0


def test_get_batch_tensors_left_pad(simple_probing_dataset, tokenizer_left_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_left_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    # Indices [0, 2] -> "This is the first example." and "Third."
    indices = [0, 2]
    batch = tokenized_dataset.get_batch_tensors(indices=indices, pad=True)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "positions" in batch

    # Cast to Tensor for shape and indexing checks
    input_ids_tensor = cast(torch.Tensor, batch["input_ids"])
    attention_mask_tensor = cast(torch.Tensor, batch["attention_mask"])

    # Shape check
    assert input_ids_tensor.shape == (2, max_length)
    assert attention_mask_tensor.shape == (2, max_length)

    # Padding check (left padding)
    assert (
        input_ids_tensor[0, 0] == tokenizer_left_pad.pad_token_id
    )  # Example 0 should have padding
    assert attention_mask_tensor[0, 0] == 0
    assert (
        input_ids_tensor[1, 0] == tokenizer_left_pad.pad_token_id
    )  # Example 2 (shorter) should have more padding
    assert attention_mask_tensor[1, 0] == 0

    # Position check (needs adjustment for left padding)
    assert "FIRST_WORD" in batch["positions"]
    # Example 0: "This" is token 0 in unpadded.
    ex0_text = simple_probing_dataset.examples[0].text
    assert ex0_text is not None  # Ensure text is not None for type checker
    ex0_pad_len = get_pad_length(ex0_text, tokenizer_left_pad, max_length, add_special)
    ex0_expected_pos = 0 + ex0_pad_len  # unpadded_pos = 0

    # Example 2: Has no "FIRST_WORD" position. Should default to 0.
    ex2_expected_pos = 0

    # Check implementation handles missing keys by defaulting to 0
    assert torch.equal(
        batch["positions"]["FIRST_WORD"],
        torch.tensor([ex0_expected_pos, ex2_expected_pos]),
    )


def test_save_load_cycle(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted
    add_special = True
    original_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_dataset")
        original_dataset.save(save_path)

        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "tokenization_config.json"))
        assert os.path.exists(os.path.join(save_path, "dataset_attributes.json"))
        assert os.path.exists(os.path.join(save_path, "hf_dataset"))

        loaded_dataset = TokenizedProbingDataset.load(save_path)

        assert isinstance(loaded_dataset, TokenizedProbingDataset)
        assert len(loaded_dataset) == len(original_dataset)
        assert (
            loaded_dataset.tokenization_config == original_dataset.tokenization_config
        )
        assert loaded_dataset.task_type == original_dataset.task_type
        assert loaded_dataset.label_mapping == original_dataset.label_mapping
        assert loaded_dataset.position_types == original_dataset.position_types

        # Deep comparison of examples
        for i in range(len(original_dataset)):
            orig_ex = cast(TokenizedProbingExample, original_dataset.examples[i])
            load_ex = cast(TokenizedProbingExample, loaded_dataset.examples[i])

            assert isinstance(load_ex, TokenizedProbingExample)
            # Compare all relevant fields
            assert load_ex.text == orig_ex.text
            assert load_ex.label == orig_ex.label
            assert load_ex.label_text == orig_ex.label_text
            assert load_ex.attributes == orig_ex.attributes
            assert load_ex.tokens == orig_ex.tokens
            assert load_ex.attention_mask == orig_ex.attention_mask

            # Compare token positions carefully (handle None vs TokenPositions({}) potentially)
            if orig_ex.token_positions is None:
                assert load_ex.token_positions is None
            else:
                assert load_ex.token_positions is not None
                assert (
                    load_ex.token_positions.positions
                    == orig_ex.token_positions.positions
                )

        # Test if loaded dataset can still produce batches
        batch = loaded_dataset.get_batch_tensors(indices=[0, 1], pad=True)
        input_ids_tensor = cast(torch.Tensor, batch["input_ids"])
        assert input_ids_tensor.shape == (2, max_length)


def test_validate_positions(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )
    assert tokenized_dataset.validate_positions()

    # Manually create an invalid position (index >= seq_len)
    ex0 = cast(TokenizedProbingExample, tokenized_dataset.examples[0])
    invalid_pos_val = len(ex0.tokens)  # Index must be < length
    invalid_ex = dataclasses.replace(
        ex0,
        token_positions=TokenPositions(
            {"FIRST_WORD": invalid_pos_val}
        ),  # Invalid position
    )
    tokenized_dataset.examples[0] = invalid_ex  # type: ignore  # Overwrite with invalid example
    assert not tokenized_dataset.validate_positions()

    # Test negative position
    ex1 = cast(TokenizedProbingExample, tokenized_dataset.examples[1])
    invalid_ex_neg = dataclasses.replace(
        ex1,
        token_positions=TokenPositions({"FIRST_WORD": -1}),  # Invalid position
    )
    tokenized_dataset.examples[1] = invalid_ex_neg  # type: ignore  # Overwrite with invalid example

    # Reset example 0 to be valid for the next check
    # Re-tokenize example 0 to ensure it's a valid TokenizedProbingExample
    valid_ex0_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=ProbingDataset(examples=[simple_probing_dataset.examples[0]]),
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        add_special_tokens=add_special,
    )
    tokenized_dataset.examples[0] = valid_ex0_dataset.examples[0]

    assert not tokenized_dataset.validate_positions()  # Example 1 is still invalid


def test_verify_position_tokens(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    verification = tokenized_dataset.verify_position_tokens(
        tokenizer=tokenizer_right_pad, position_key="FIRST_WORD"
    )

    # verification maps example index to results dict
    assert isinstance(verification, dict)
    assert 0 in verification  # Example 0 has FIRST_WORD
    assert 1 in verification  # Example 1 has FIRST_WORD
    assert 2 not in verification  # Example 2 does not have FIRST_WORD

    # Check example 0
    verification_0 = cast(dict, verification.get(0, {}))
    assert "FIRST_WORD" in verification_0
    res0 = verification_0["FIRST_WORD"]
    assert res0["position"] == 0  # Corrected position
    assert res0["token_text"] == "This"  # Corrected expected token
    assert "error" not in res0

    # Check example 1
    verification_1 = cast(dict, verification.get(1, {}))
    assert "FIRST_WORD" in verification_1
    res1 = verification_1["FIRST_WORD"]
    assert res1["position"] == 0  # Corrected position
    assert res1["token_text"] == "A"  # Corrected expected token for " A"
    assert "error" not in res1


def test_show_token_context_right_pad(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    context = tokenized_dataset.show_token_context(
        example_idx=0,
        position_key="FIRST_WORD",
        tokenizer=tokenizer_right_pad,
        context_size=2,
    )

    assert context["example_idx"] == 0
    assert context["position_key"] == "FIRST_WORD"
    assert context["token_count"] == max_length
    assert context["padding_side"] == "right"
    assert "results" in context

    res = context["results"]  # Assuming single position
    assert res["position"] == 0  # Corrected position (0)
    assert res["original_position"] == 0  # Corrected original position (0)
    assert res["token_info"]["text"] == "This"  # Corrected expected token
    assert res["context_window"]["start"] == 0
    assert res["context_window"]["end"] == 3  # pos 0, context 2 -> indices 0, 1, 2
    # GPT2: This is the ... (No BOS)
    ex0_text = simple_probing_dataset.examples[0].text
    assert ex0_text is not None
    # Get first 3 tokens after tokenization (no special tokens assumed at start)
    expected_context_tokens = tokenizer_right_pad.encode(
        ex0_text, add_special_tokens=False, padding=False, truncation=False
    )[:3]
    expected_texts = [tokenizer_right_pad.decode([t]) for t in expected_context_tokens]
    assert res["context_window"]["tokens"] == expected_context_tokens
    assert res["context_window"]["texts"] == expected_texts
    # Check marked context correctly identifies "This"
    marked_token_text = expected_texts[0]  # Position 0
    assert f"[[ {marked_token_text} ]]" in res["context_window"]["marked_context"]


def test_show_token_context_left_pad(simple_probing_dataset, tokenizer_left_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_left_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    context = tokenized_dataset.show_token_context(
        example_idx=0,  # "This is the first example."
        position_key="FIRST_WORD",
        tokenizer=tokenizer_left_pad,
        context_size=2,
    )

    assert context["padding_side"] == "left"
    res = context["results"]

    # Calculate expected padded position
    ex0_text = simple_probing_dataset.examples[0].text
    assert ex0_text is not None
    pad_len = get_pad_length(ex0_text, tokenizer_left_pad, max_length, add_special)
    expected_padded_pos = 0 + pad_len  # Unpadded position 0 ("This") + padding

    assert res["position"] == expected_padded_pos  # Position within the padded sequence
    assert (
        res["original_position"] == 0
    )  # Position relative to unpadded real tokens (position 0)
    assert res["token_info"]["text"] == "This"  # Corrected expected token

    # Check context window indices are correct relative to padded sequence
    expected_start = max(0, expected_padded_pos - 2)
    expected_end = min(max_length, expected_padded_pos + 3)  # pos + context + 1
    assert res["context_window"]["start"] == expected_start
    assert res["context_window"]["end"] == expected_end
    assert "[[ This ]]" in res["context_window"]["marked_context"]


def test_verify_padding_left(simple_probing_dataset, tokenizer_left_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_left_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    verification = tokenized_dataset.verify_padding(
        tokenizer=tokenizer_left_pad,
        max_length=max_length,
        examples_to_check=2,  # Check first 2 examples
    )

    assert verification["padding_side"] == "left"
    assert verification["max_length"] == max_length
    assert 0 in verification["examples"]
    assert 1 in verification["examples"]

    # Example 0: "This is the first example." -> pos 1 ("This")
    ex0_text = simple_probing_dataset.examples[0].text
    assert ex0_text is not None
    ex0_res = verification["examples"][0]["positions"]["FIRST_WORD"]
    assert ex0_res["position_matches"] is True
    assert ex0_res["token_text"] == "This"
    ex0_pad_len = get_pad_length(ex0_text, tokenizer_left_pad, max_length, add_special)
    # Calculate expected position (unpadded pos 0 + padding)
    ex0_expected_calc = 0 + ex0_pad_len
    assert ex0_res["expected_position"] == ex0_expected_calc
    # Check padded vs expected
    assert ex0_res["padded_position"] == ex0_res["expected_position"]

    # Example 1: "A second, longer example here." -> pos 0 ("A")
    ex1_text = simple_probing_dataset.examples[1].text
    assert ex1_text is not None
    ex1_res = verification["examples"][1]["positions"]["FIRST_WORD"]
    assert ex1_res["position_matches"] is True
    assert ex1_res["token_text"] == "A"
    ex1_pad_len = get_pad_length(ex1_text, tokenizer_left_pad, max_length, add_special)
    ex1_expected_calc = 0 + ex1_pad_len
    assert ex1_res["expected_position"] == ex1_expected_calc
    assert ex1_res["padded_position"] == ex1_res["expected_position"]


def test_verify_padding_right(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    verification = tokenized_dataset.verify_padding(
        tokenizer=tokenizer_right_pad, max_length=max_length, examples_to_check=2
    )

    assert verification["padding_side"] == "right"

    # Example 0: "This" -> pos 1
    ex0_res = verification["examples"][0]["positions"]["FIRST_WORD"]
    assert ex0_res["position_matches"] is True
    assert ex0_res["token_text"] == "This"
    assert (
        ex0_res["expected_position"] == 0
    )  # Corrected expected position (unpadded pos = 0)
    assert ex0_res["padded_position"] == 0

    # Example 1: "A" -> pos 0
    ex1_res = verification["examples"][1]["positions"]["FIRST_WORD"]
    assert ex1_res["position_matches"] is True
    assert ex1_res["token_text"] == "A"
    assert ex1_res["expected_position"] == 0  # Corrected expected position
    assert ex1_res["padded_position"] == 0


def test_to_hf_dataset(simple_probing_dataset, tokenizer_right_pad):
    max_length = 16  # Adjusted
    add_special = True
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=simple_probing_dataset,
        tokenizer=tokenizer_right_pad,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special,
        return_tensors=None,
    )

    hf_dataset = tokenized_dataset._to_hf_dataset()
    assert isinstance(hf_dataset, datasets.Dataset)

    # Check basic columns from ProbingDataset
    assert "text" in hf_dataset.column_names
    assert "label" in hf_dataset.column_names
    assert "label_text" in hf_dataset.column_names
    # Check if 'id' attribute was flattened (depends on ProbingDataset._to_hf_dataset impl.)
    # assert "attribute_id" in hf_dataset.column_names # Example if flattened

    # Check columns added by TokenizedProbingDataset
    assert "tokens" in hf_dataset.column_names
    assert "attention_mask" in hf_dataset.column_names
    assert "token_pos_FIRST_WORD" in hf_dataset.column_names  # Specific to this dataset

    # Verify content
    assert len(hf_dataset) == 3
    assert hf_dataset[0]["text"] == simple_probing_dataset.examples[0].text
    assert hf_dataset[0]["label"] == simple_probing_dataset.examples[0].label
    assert len(hf_dataset[0]["tokens"]) == max_length
    assert hf_dataset[0]["token_pos_FIRST_WORD"] == 0  # Corrected position
    assert hf_dataset[1]["token_pos_FIRST_WORD"] == 0  # Corrected position
    assert (
        hf_dataset[2]["token_pos_FIRST_WORD"] is None
    )  # Example 2, no position defined


# TODO: Add tests for edge cases like empty datasets, examples with no text, etc.
