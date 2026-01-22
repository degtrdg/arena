"""
Parse ai-psychosis transcript files into structured data.
"""

import re
from pathlib import Path
from typing import Optional


def parse_filename(filepath: str | Path) -> dict:
    """
    Extract metadata from transcript filename.

    Filename format: {Character}_{Model}_{Timestamp}_target.md

    Examples:
        Adrian_anthropic-claude-sonnet-4-20250514_20250819_081336_target.md
        Colin_openrouter-google-gemini-2.5-pro_20250819_081336_target.md
    """
    filepath = Path(filepath)
    name = filepath.stem  # Remove .md

    # Remove _target suffix
    if name.endswith("_target"):
        name = name[:-7]

    parts = name.split("_")

    # Character is first part
    character = parts[0]

    # Timestamp is last two parts (date_time)
    timestamp = f"{parts[-2]}_{parts[-1]}"

    # Model is everything in between
    model = "_".join(parts[1:-2])

    return {
        "character": character,
        "model": model,
        "timestamp": timestamp,
        "filepath": str(filepath),
    }


def normalize_model_name(name: str) -> str:
    """
    Normalize model name for matching between transcripts and CSVs.

    Transcript format: anthropic-claude-sonnet-4-20250514
    CSV format: anthropic/claude-sonnet-4-20250514

    Returns lowercase with slashes replaced by hyphens.
    """
    return name.lower().replace("/", "-").replace("_", "-")


def parse_markdown_transcript(filepath: str | Path) -> list[dict]:
    """
    Parse a _target.md file into structured conversation turns.

    Returns:
        List of dicts with keys:
        - 'turn_number': int (0-indexed)
        - 'user_message': str
        - 'assistant_response': str
        - 'character': str
        - 'model': str
        - 'timestamp': str
    """
    filepath = Path(filepath)
    metadata = parse_filename(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    turns = []
    current_turn = {"user_message": None, "assistant_response": None}
    turn_number = 0

    # Split by the --- delimiter
    sections = re.split(r"\n---\n", content)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if this is a user message
        if section.startswith("### 👤 User"):
            # Extract message after the header
            lines = section.split("\n", 1)
            if len(lines) > 1:
                current_turn["user_message"] = lines[1].strip()

        # Check if this is an assistant response
        elif section.startswith("### 🤖 Assistant"):
            # Extract response after the header
            lines = section.split("\n", 1)
            if len(lines) > 1:
                current_turn["assistant_response"] = lines[1].strip()

            # If we have both user and assistant, save the turn
            if current_turn["user_message"] and current_turn["assistant_response"]:
                turns.append({
                    "turn_number": turn_number,
                    "user_message": current_turn["user_message"],
                    "assistant_response": current_turn["assistant_response"],
                    "character": metadata["character"],
                    "model": metadata["model"],
                    "model_normalized": normalize_model_name(metadata["model"]),
                    "timestamp": metadata["timestamp"],
                })
                turn_number += 1
                current_turn = {"user_message": None, "assistant_response": None}

    return turns


def load_all_transcripts(transcripts_dir: str | Path) -> list[dict]:
    """
    Load all _target.md transcripts from a directory.

    Returns:
        List of all conversation turns from all files.
    """
    transcripts_dir = Path(transcripts_dir)
    all_turns = []

    for filepath in transcripts_dir.glob("*_target.md"):
        try:
            turns = parse_markdown_transcript(filepath)
            all_turns.extend(turns)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

    print(f"Loaded {len(all_turns)} turns from {len(list(transcripts_dir.glob('*_target.md')))} files")
    return all_turns


if __name__ == "__main__":
    # Test parsing
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "ai-psychosis/full_transcripts/Adrian_anthropic-claude-sonnet-4-20250514_20250819_081336_target.md"

    turns = parse_markdown_transcript(filepath)
    print(f"Parsed {len(turns)} turns")
    for turn in turns[:2]:
        print(f"\n--- Turn {turn['turn_number']} ---")
        print(f"Character: {turn['character']}, Model: {turn['model']}")
        print(f"User: {turn['user_message'][:100]}...")
        print(f"Assistant: {turn['assistant_response'][:100]}...")
