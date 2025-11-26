"""OpenAI-compatible API client utilities."""

import json
import logging
import time
from typing import Optional, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


def get_chat_response(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    return_raw: bool = False,
) -> Dict[str, Any]:
    """Get response from OpenAI-compatible API.

    Args:
        client: OpenAI client instance
        prompt: User prompt to send
        model: Model name to use
        temperature: Sampling temperature (optional)
        max_tokens: Maximum tokens to generate (optional)
        top_p: Nucleus sampling probability (optional)
        frequency_penalty: Frequency penalty (optional)
        presence_penalty: Presence penalty (optional)
        return_raw: If True, include raw request/response in result

    Returns:
        Dict with keys:
            - content: Generated text content
            - reasoning_content: Reasoning trace if available
            - stop_reason: Finish reason (stop, length, etc.)
            - raw: Raw request/response data (only if return_raw=True)
    """
    logger.debug(f"API Call - Model: {model}")
    logger.debug(f"API Call - Temperature: {temperature}, Max tokens: {max_tokens}")
    logger.debug(
        f"API Call - Prompt: {prompt[:200]}..."
        if len(prompt) > 200
        else f"API Call - Prompt: {prompt}"
    )

    # Build kwargs dict, only including non-None values
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        kwargs["top_p"] = top_p
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty

    response = client.chat.completions.create(**kwargs)

    message = response.choices[0].message
    content = message.content
    reasoning_content = getattr(message, "reasoning_content", None)
    stop_reason = response.choices[0].finish_reason

    # Log warning if content is None
    if content is None:
        logger.warning(
            f"API response returned None content. Stop reason: {stop_reason}"
        )

    logger.debug(
        f"API Response: {content[:200]}..."
        if content and len(content) > 200
        else f"API Response: {content}"
    )
    logger.debug(
        f"API Call - Tokens used: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}"
    )

    result = {
        "content": content,
        "reasoning_content": reasoning_content,
        "stop_reason": stop_reason,
    }

    # Add raw request/response data if requested
    if return_raw:
        result["raw"] = {
            "request": kwargs,
            "response": response.model_dump()
            if hasattr(response, "model_dump")
            else response.dict(),
        }

    return result


def save_raw_api_call(file_path: str, row_index: int, result: Dict[str, Any]) -> None:
    """Append raw API request/response to JSONL file.

    Args:
        file_path: Path to JSONL file
        row_index: Index of the row being processed
        result: Result dict containing raw API data (must have 'raw' key)
    """
    if "raw" not in result:
        return

    record = {
        "timestamp": time.time(),
        "row_index": row_index,
        "request": result["raw"]["request"],
        "response": result["raw"]["response"],
    }

    # Append to JSONL file (one JSON object per line)
    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")
