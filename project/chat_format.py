"""
Chat template formatting utilities.

Handles prompt formatting for different model families (Gemma, ChatML/Dolphin, etc.).
Uses the tokenizer's built-in apply_chat_template when available, with manual fallbacks.
"""

from typing import Optional, List, Dict


def format_chat_prompt(
    user_message: str,
    assistant_response: str,
    model_name: str,
    tokenizer=None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Format a user-assistant exchange in the model's expected chat template.

    Args:
        user_message: The user's input/question
        assistant_response: The assistant's response
        model_name: Model name/path to determine format
        tokenizer: Optional tokenizer (uses apply_chat_template if available)
        system_prompt: Optional system prompt

    Returns:
        Formatted string in the model's chat format
    """
    messages = _build_messages(user_message, assistant_response, system_prompt)

    # Try tokenizer's built-in chat template first
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass  # Fall through to manual formatting

    # Manual fallback based on model name
    model_lower = model_name.lower()

    if "dolphin" in model_lower or "chatml" in model_lower or "mixtral" in model_lower:
        return _format_chatml(messages)
    elif "gemma" in model_lower:
        return _format_gemma(messages)
    else:
        # Generic fallback
        return _format_chatml(messages)


def _build_messages(
    user_message: str,
    assistant_response: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build a messages list from components."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_response})
    return messages


def _format_chatml(messages: List[Dict[str, str]]) -> str:
    """Format messages in ChatML format (used by Dolphin, OpenHermes, etc.)."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts)


def _format_gemma(messages: List[Dict[str, str]]) -> str:
    """Format messages in Gemma's chat format."""
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            # Gemma doesn't have a system turn; prepend to next user message
            continue
        elif msg["role"] == "user":
            # If there was a system message, prepend it
            system_msgs = [m for m in messages if m["role"] == "system"]
            content = msg["content"]
            if system_msgs:
                content = system_msgs[0]["content"] + "\n\n" + content
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif msg["role"] == "assistant":
            parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>")
    return "\n".join(parts)
