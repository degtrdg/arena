"""
Build probity-compatible datasets for sycophancy probe training.
"""

import sys
from pathlib import Path

# Add probity to path
sys.path.insert(0, str(Path(__file__).parent.parent / "probity"))

from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.datasets.position_finder import Position, PositionFinder
from transformers import AutoTokenizer


def create_probing_examples(labeled_data: list[dict]) -> list[ProbingExample]:
    """
    Convert labeled transcript data to ProbingExamples.

    Args:
        labeled_data: List of dicts from get_labeled_data()

    Returns:
        List of ProbingExample objects for probity.
    """
    examples = []

    # Position finder for end of text (last non-whitespace character)
    end_finder = PositionFinder.from_regex(r"[\S].{0,1}$")

    for item in labeled_data:
        # Skip items without labels
        if item.get("sycophancy_label") is None:
            continue

        text = item["assistant_response"]

        # Skip very short responses
        if len(text) < 20:
            continue

        # Find end position
        positions_dict = {}
        try:
            end_pos = end_finder(text)
            if end_pos:
                positions_dict["END_POSITION"] = end_pos[0]
        except Exception:
            # Fallback: use last character position
            positions_dict["END_POSITION"] = Position(start=len(text) - 2, end=len(text))

        # Create label
        label = int(item["sycophancy_label"])
        label_text = "sycophantic" if label == 1 else "non_sycophantic"

        examples.append(ProbingExample(
            text=text,
            label=label,
            label_text=label_text,
            character_positions=CharacterPositions(positions_dict) if positions_dict else None,
            attributes={
                "character": item["character"],
                "model": item["model"],
                "turn_number": item["turn_number"],
                "pushback_rating": item.get("pushback_rating"),
                "delusion_confirmation_rating": item.get("delusion_confirmation_rating"),
            }
        ))

    print(f"Created {len(examples)} ProbingExamples")
    return examples


def build_probing_dataset(labeled_data: list[dict]) -> ProbingDataset:
    """
    Build a ProbingDataset from labeled transcript data.

    Args:
        labeled_data: List of dicts from get_labeled_data()

    Returns:
        ProbingDataset ready for tokenization.
    """
    examples = create_probing_examples(labeled_data)

    dataset = ProbingDataset(
        examples=examples,
        task_type="classification",
        label_mapping={"non_sycophantic": 0, "sycophantic": 1},
        dataset_attributes={
            "description": "Sycophancy detection from ai-psychosis red-teaming conversations",
            "source": "ai-psychosis",
        }
    )

    return dataset


def tokenize_dataset(
    probing_dataset: ProbingDataset,
    model_name: str = "google/gemma-2-2b",
    max_length: int = 1024
) -> TokenizedProbingDataset:
    """
    Tokenize a ProbingDataset for a specific model.

    Args:
        probing_dataset: The ProbingDataset to tokenize
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length (truncates longer sequences)

    Returns:
        TokenizedProbingDataset ready for activation collection.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    # Print stats
    token_lengths = [len(ex.tokens) for ex in tokenized_dataset.examples]
    print(f"Tokenized {len(tokenized_dataset.examples)} examples")
    print(f"Token length stats: min={min(token_lengths)}, max={max(token_lengths)}, mean={sum(token_lengths)/len(token_lengths):.1f}")

    return tokenized_dataset


def build_full_dataset(
    transcripts_dir: str,
    grades_dir: str,
    model_name: str = "google/gemma-2-2b",
    max_length: int = 1024
) -> TokenizedProbingDataset:
    """
    Full pipeline: load data, label it, build dataset, tokenize.

    Args:
        transcripts_dir: Path to full_transcripts directory
        grades_dir: Path to result_grades directory
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length

    Returns:
        TokenizedProbingDataset ready for probe training.
    """
    from .labeling import get_labeled_data

    # Load and label data
    labeled_data = get_labeled_data(transcripts_dir, grades_dir, only_labeled=True)

    # Build probing dataset
    probing_dataset = build_probing_dataset(labeled_data)

    # Tokenize
    tokenized_dataset = tokenize_dataset(probing_dataset, model_name, max_length)

    return tokenized_dataset


if __name__ == "__main__":
    # Test dataset building
    dataset = build_full_dataset(
        "ai-psychosis/full_transcripts",
        "ai-psychosis/result_grades",
        model_name="google/gemma-2-2b",
        max_length=512  # Use shorter for testing
    )

    print(f"\nDataset size: {len(dataset.examples)}")
    print(f"Position types: {dataset.position_types}")

    # Show a sample
    ex = dataset.examples[0]
    print(f"\nSample example:")
    print(f"  Text: {ex.text[:100]}...")
    print(f"  Label: {ex.label}")
    print(f"  Tokens: {ex.tokens[:10]}...")
