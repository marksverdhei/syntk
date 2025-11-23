"""API interaction utilities for syntk.

Provides functions for interacting with OpenAI-compatible APIs.
"""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


def get_chat_response(
    client: OpenAI, prompt: str, api_args, gen_args
) -> str:
    """Get response from OpenAI-compatible API.
    
    Args:
        client: OpenAI client instance
        prompt: The prompt text to send
        api_args: APIArguments with model and base_url
        gen_args: GenerationArguments with temperature, max_tokens, etc.
        
    Returns:
        The response text from the API
    """
    logger.debug(f"API Call - Model: {api_args.model}")
    logger.debug(
        f"API Call - Temperature: {gen_args.temperature}, Max tokens: {gen_args.max_tokens}"
    )
    logger.debug(
        f"API Call - Prompt: {prompt[:200]}..."
        if len(prompt) > 200
        else f"API Call - Prompt: {prompt}"
    )

    # Build kwargs dict, only including non-None values
    kwargs = {
        "model": api_args.model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if gen_args.temperature is not None:
        kwargs["temperature"] = gen_args.temperature
    if gen_args.max_tokens is not None:
        kwargs["max_tokens"] = gen_args.max_tokens
    if gen_args.top_p is not None:
        kwargs["top_p"] = gen_args.top_p
    if gen_args.frequency_penalty is not None:
        kwargs["frequency_penalty"] = gen_args.frequency_penalty
    if gen_args.presence_penalty is not None:
        kwargs["presence_penalty"] = gen_args.presence_penalty

    response = client.chat.completions.create(**kwargs)

    response_text = response.choices[0].message.content
    logger.debug(
        f"API Response: {response_text[:200]}..."
        if len(response_text) > 200
        else f"API Response: {response_text}"
    )
    logger.debug(
        f"API Call - Tokens used: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}"
    )

    return response_text
