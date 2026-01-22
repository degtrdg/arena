import pytest
from transformers import AutoTokenizer

# Assuming Position and PositionFinder are in probity.datasets.position_finder
from probity.datasets.position_finder import Position, PositionFinder

# --- Fixtures ---


@pytest.fixture(scope="module")
def tokenizer():
    """Provides a tokenizer instance for tests."""
    # Using a simple tokenizer like gpt2 for testing purposes
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Standard practice
    return tokenizer


# --- Test Position Class ---


def test_position_initialization():
    """Test valid Position initialization."""
    pos1 = Position(start=5)
    assert pos1.start == 5
    assert pos1.end is None

    pos2 = Position(start=10, end=15)
    assert pos2.start == 10
    assert pos2.end == 15


def test_position_invalid_start():
    """Test Position validation for negative start."""
    with pytest.raises(ValueError, match="Start position must be non-negative"):
        Position(start=-1)


def test_position_invalid_end():
    """Test Position validation for end < start."""
    with pytest.raises(
        ValueError, match="End position must be greater than start position"
    ):
        Position(start=10, end=5)


# --- Test PositionFinder Static Methods ---


# Test from_template
def test_positionfinder_from_template_simple():
    """Test finding position from a simple template."""
    template = "The color is {COLOR}."
    marker = "{COLOR}"
    finder = PositionFinder.from_template(template, marker)
    text = "The color is blue."
    position = finder(text)
    assert isinstance(position, Position)
    assert position.start == 13  # Start of 'blue'
    assert position.end == 17  # End of 'blue'
    assert text[position.start : position.end] == "blue"


def test_positionfinder_from_template_complex():
    """Test finding position from a template with multiple variables."""
    template = "{PERSON} likes {FOOD}."
    marker = "{PERSON}"
    finder = PositionFinder.from_template(template, marker)
    text = "Alice likes pizza."
    position = finder(text)
    assert position.start == 0  # Start of 'Alice'
    assert position.end == 5  # End of 'Alice'
    assert text[position.start : position.end] == "Alice"

    marker_food = "{FOOD}"
    finder_food = PositionFinder.from_template(template, marker_food)
    position_food = finder_food(text)
    assert position_food.start == 12  # Start of 'pizza'
    assert position_food.end == 17  # End of 'pizza'
    assert text[position_food.start : position_food.end] == "pizza"


def test_positionfinder_from_template_no_match():
    """Test template finder when text doesn't match."""
    template = "Value: {VALUE}"
    marker = "{VALUE}"
    finder = PositionFinder.from_template(template, marker)
    text = "This is different"
    with pytest.raises(ValueError, match="Text does not match template"):
        finder(text)


# Test from_regex
def test_positionfinder_from_regex_single_match():
    """Test finding a single regex match."""
    pattern = r"\b\d{3}\b"  # Find 3-digit numbers
    finder = PositionFinder.from_regex(pattern)
    text = "Agent 007 is here."
    positions = finder(text)
    assert len(positions) == 1
    assert positions[0].start == 6
    assert positions[0].end == 9
    assert text[positions[0].start : positions[0].end] == "007"


def test_positionfinder_from_regex_multiple_matches():
    """Test finding multiple regex matches."""
    pattern = r"cat"
    finder = PositionFinder.from_regex(pattern)
    text = "The cat sat on the catalog."
    positions = finder(text)
    assert len(positions) == 2
    assert positions[0].start == 4
    assert positions[0].end == 7
    assert text[positions[0].start : positions[0].end] == "cat"
    assert positions[1].start == 19
    assert positions[1].end == 22
    assert text[positions[1].start : positions[1].end] == "cat"


def test_positionfinder_from_regex_capture_group():
    """Test finding a specific capture group with regex."""
    pattern = r"Name: (\w+)"
    finder = PositionFinder.from_regex(pattern, group=1)  # Capture group 1
    text = "ID: 123 Name: Bob Age: 30"
    positions = finder(text)
    assert len(positions) == 1
    assert positions[0].start == 14  # Start of 'Bob'
    assert positions[0].end == 17  # End of 'Bob'
    assert text[positions[0].start : positions[0].end] == "Bob"


def test_positionfinder_from_regex_no_match():
    """Test regex finder when pattern doesn't match."""
    pattern = r"xyz"
    finder = PositionFinder.from_regex(pattern)
    text = "abc def ghi"
    positions = finder(text)
    assert len(positions) == 0


# Test from_char_position
def test_positionfinder_from_char_position_valid():
    """Test finding a valid fixed character position."""
    finder = PositionFinder.from_char_position(5)
    text = "0123456789"
    position = finder(text)
    assert isinstance(position, Position)
    assert position.start == 5
    assert position.end is None  # Single character position


def test_positionfinder_from_char_position_invalid():
    """Test finding an out-of-bounds character position."""
    finder = PositionFinder.from_char_position(10)
    text = "0123456789"  # Length 10
    with pytest.raises(ValueError, match="Position 10 is beyond text length 10"):
        finder(text)

    finder_long = PositionFinder.from_char_position(15)
    with pytest.raises(ValueError, match="Position 15 is beyond text length 10"):
        finder_long(text)


# Test convert_to_token_position
def test_convert_to_token_position_no_special(tokenizer):
    """Test converting char pos to token pos without special tokens."""
    text = "This is a test sentence."
    # Tokenization: ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ġsentence', '.']
    # Offsets:    [(0, 4), (4, 7), (7, 9), (9, 14), (14, 23), (23, 24)]

    # Position within "test"
    char_pos = Position(start=10)  # Character 'e' in 'test'
    token_idx = PositionFinder.convert_to_token_position(
        char_pos, text, tokenizer, add_special_tokens=False
    )
    assert token_idx == 3  # Index of 'Ġtest'

    # Position at the start of "is"
    char_pos_start = Position(start=5)  # Character 'i' in 'is'
    token_idx_start = PositionFinder.convert_to_token_position(
        char_pos_start, text, tokenizer, add_special_tokens=False
    )
    assert token_idx_start == 1  # Index of 'Ġis'


def test_convert_to_token_position_with_special_bos(tokenizer):
    """Test converting char pos to token pos with BOS special token."""
    text = "Hello world."
    # Assuming GPT2 adds BOS: ['<|endoftext|>', 'Hello', 'Ġworld', '.']
    # Clean Tokens: ['Hello', 'Ġworld', '.']
    # Clean Offsets: [(0, 5), (5, 11), (11, 12)]

    # Position within "world"
    char_pos = Position(start=7)  # Character 'o' in 'world'

    # Find clean index first (should be 1 for 'Ġworld')
    clean_encoding = tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=False
    )
    clean_token_idx_check = -1
    for idx, (start, end) in enumerate(clean_encoding.offset_mapping):
        if start <= char_pos.start < end:
            clean_token_idx_check = idx
            break
    assert clean_token_idx_check == 1

    # Now convert with special tokens
    token_idx = PositionFinder.convert_to_token_position(
        char_pos, text, tokenizer, add_special_tokens=True
    )

    # Expecting index 2: ['<|endoftext|>', 'Hello', 'Ġworld', '.']
    # Check if tokenizer actually adds BOS/EOS. GPT2 uses EOS as BOS if not specified.
    prefix_tokens = 0
    if getattr(tokenizer, "bos_token_id", None) is not None:
        prefix_tokens += 1
    elif getattr(tokenizer, "cls_token_id", None) is not None:
        prefix_tokens += 1
    # GPT2 uses eos_token as bos_token if bos_token is not set
    elif (
        getattr(tokenizer, "eos_token_id", None) is not None
        and getattr(tokenizer, "bos_token_id", None) is None
    ):
        prefix_tokens += 1

    expected_token_idx = clean_token_idx_check + prefix_tokens
    assert token_idx == expected_token_idx


def test_convert_to_token_position_char_not_aligned(tokenizer):
    """Test conversion when char pos doesn't align with token boundaries."""
    text = "Example sentence."
    # Example: Sometimes a space might not be part of a token offset map
    # This test is slightly contrived as tokenizers usually cover all chars.
    # We simulate by choosing a position *between* tokens if possible,
    # although `char_offset_mapping` usually prevents this exact scenario.
    # Let's test the error raising if the loop completes without finding a match.

    # Use a position clearly outside any valid token span
    # Mock the offset mapping to ensure no match
    class MockTokenizer:
        def __call__(
            self, text, return_offsets_mapping=False, add_special_tokens=False
        ):
            if add_special_tokens is False and return_offsets_mapping:
                # Simulate offsets that *don't* cover position 5
                return {
                    "input_ids": [0, 1, 2],
                    "offset_mapping": [(0, 2), (3, 5), (6, 8)],
                }
            elif add_special_tokens is True:
                # Simulate adding a BOS token
                return {"input_ids": [100, 0, 1, 2]}
            else:
                return {"input_ids": [0, 1, 2]}

        bos_token_id = 100
        padding_side = "right"

    mock_tokenizer = MockTokenizer()
    char_pos = Position(
        start=5
    )  # Position 5 is not covered by [(0, 2), (3, 5), (6, 8)]

    with pytest.raises(
        ValueError, match="Character position 5 not aligned with any token offset."
    ):
        PositionFinder.convert_to_token_position(
            char_pos, text, mock_tokenizer, add_special_tokens=False  # type: ignore
        )

    with pytest.raises(
        ValueError, match="Character position 5 not aligned with any token offset."
    ):
        PositionFinder.convert_to_token_position(
            char_pos, text, mock_tokenizer, add_special_tokens=True  # type: ignore
        )


# Test validate_token_position
def test_validate_token_position_valid():
    """Test validating valid token positions."""
    tokens = [101, 1000, 2000, 3000, 102]  # Length 5
    assert PositionFinder.validate_token_position(0, tokens) is True  # Start
    assert PositionFinder.validate_token_position(2, tokens) is True  # Middle
    assert PositionFinder.validate_token_position(4, tokens) is True  # End - 1


def test_validate_token_position_invalid():
    """Test validating invalid token positions."""
    tokens = [101, 1000, 2000, 3000, 102]  # Length 5
    assert PositionFinder.validate_token_position(-1, tokens) is False  # Negative
    assert PositionFinder.validate_token_position(5, tokens) is False  # Equal to length
    assert (
        PositionFinder.validate_token_position(6, tokens) is False
    )  # Greater than length
    assert PositionFinder.validate_token_position(0, []) is False  # Empty list
