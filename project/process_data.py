"""
Process humor_samples.csv into JSON format for SFT training.

The CSV contains multi-turn conversations where we want to train the model
to generate the 'Golden Response' (funny version) instead of the original
final assistant response.
"""

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def parse_conversation(conversation_text: str) -> List[Dict[str, str]]:
    """
    Parse a conversation string into a list of message dicts.

    Format: "User: message\nAssistant: message\nUser: message..."
    Returns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    messages = []

    # Split by role markers, keeping the marker
    # Pattern matches "User:" or "Assistant:" at start of line or after newline
    pattern = r'(?:^|\n)(User|Assistant):\s*'

    parts = re.split(pattern, conversation_text.strip())

    # parts will be: ['', 'User', 'message...', 'Assistant', 'message...', ...]
    # Skip the first empty element, then pair up role and content
    i = 1
    while i < len(parts) - 1:
        role = parts[i].lower()
        content = parts[i + 1].strip()

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

        i += 2

    return messages


def parse_golden_response(golden_text: str) -> Tuple[str, str]:
    """
    Parse the golden response which includes the user's final message and the funny response.

    Format: "User: final question\nFunny Assistant: funny response"
    Returns: (user_message, funny_response)
    """
    # Split on "Funny Assistant:" to get the funny response
    if "Funny Assistant:" in golden_text:
        parts = golden_text.split("Funny Assistant:", 1)
        user_part = parts[0].strip()
        funny_response = parts[1].strip()

        # Extract just the user message from user_part
        if user_part.startswith("User:"):
            user_message = user_part[5:].strip()
        else:
            user_message = user_part

        return user_message, funny_response

    return "", golden_text


def process_csv_to_json(csv_path: str, output_path: str) -> List[Dict]:
    """
    Process the humor samples CSV into JSON format for training.

    For each sample, we create a training example where:
    - The conversation history (up to the final user message) is preserved
    - The final assistant response is replaced with the Golden Response (funny version)
    """
    samples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                sample_id = row.get('ID', '')
                joke_id = row.get('Joke ID', '')
                original_conv = row.get('Original Conversation', '')
                golden_response = row.get('Golden Response', '')
                opening = row.get('Opening', '')
                rubric = row.get('Rubric', '')

                if not original_conv or not golden_response:
                    print(f"Skipping {sample_id}: missing conversation or golden response")
                    continue

                # Parse the original conversation
                messages = parse_conversation(original_conv)

                if len(messages) < 2:
                    print(f"Skipping {sample_id}: conversation too short")
                    continue

                # Parse the golden response to get BOTH the final user message and funny response
                user_message, funny_response = parse_golden_response(golden_response)

                # Build the training example:
                # Since the CSV may have truncated conversations, we need to be defensive
                # Strategy: Keep only complete, well-formed user-assistant pairs, then add golden pair

                conversation_history = []

                # Build history from complete pairs only, stopping at any irregularity
                # BUT exclude the final pair since we'll replace it with the golden response
                i = 0
                while i < len(messages) - 2:  # -2 to leave room for at least one pair to replace
                    # Check if we have a proper user-assistant pair
                    if (i + 1 < len(messages) and
                        messages[i]["role"] == "user" and
                        messages[i+1]["role"] == "assistant"):
                        # Add this complete pair
                        conversation_history.append(messages[i])
                        conversation_history.append(messages[i+1])
                        i += 2
                    else:
                        # Hit an irregularity, stop processing
                        break

                # Now add the golden user message and funny response
                # (This replaces the final user-assistant pair from the original)
                if user_message:
                    conversation_history.append({"role": "user", "content": user_message})
                conversation_history.append({"role": "assistant", "content": funny_response})

                # Final validation: ensure proper alternation
                valid = True
                for j in range(len(conversation_history)):
                    if j == 0 and conversation_history[j]["role"] != "user":
                        valid = False
                        break
                    if j > 0:
                        prev_role = conversation_history[j-1]["role"]
                        curr_role = conversation_history[j]["role"]
                        if prev_role == curr_role:
                            valid = False
                            break

                if not valid:
                    print(f"Skipping {sample_id}: final validation failed")
                    continue

                # Output just the messages array - that's all we need for SFT
                sample = {
                    "id": sample_id,
                    "messages": conversation_history
                }

                samples.append(sample)

            except Exception as e:
                print(f"Error processing row: {e}")
                continue

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed {len(samples)} samples")
    print(f"Saved to {output_path}")

    return samples


def preview_samples(samples: List[Dict], n: int = 3):
    """Preview a few processed samples"""
    print("\n" + "="*60)
    print("Sample Preview")
    print("="*60)

    for i, sample in enumerate(samples[:n]):
        print(f"\n--- Sample {i+1} (ID: {sample['id']}) ---")
        print("Messages:")
        for msg in sample["messages"]:
            role = msg["role"].capitalize()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"  {role}: {content}")
        print()


if __name__ == "__main__":
    csv_path = "data/humor_samples.csv"
    output_path = "data/humor_dataset.json"

    print("Processing humor samples CSV to JSON...")
    samples = process_csv_to_json(csv_path, output_path)

    if samples:
        preview_samples(samples)
