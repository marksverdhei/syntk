"""Tests for syntk.api_client.get_chat_response."""

from unittest.mock import MagicMock
import logging

import pytest
from openai import AuthenticationError

from syntk.api_client import get_chat_response


def _make_response(
    content="hello",
    reasoning_content=None,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
    total_tokens=15,
    include_model_dump=True,
):
    message = MagicMock()
    message.content = content
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    else:
        # getattr falls back to AttributeError, simulating no attribute
        del message.reasoning_content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    if include_model_dump:
        response.model_dump.return_value = {"id": "resp-1"}

    return response


def _make_client(response=None):
    client = MagicMock()
    if response is None:
        response = _make_response()
    client.chat.completions.create.return_value = response
    return client


class TestGetChatResponse:
    def test_returns_dict(self):
        client = _make_client()
        result = get_chat_response(client, "hello", "gpt-4")
        assert isinstance(result, dict)

    def test_content_in_result(self):
        response = _make_response(content="the answer")
        client = _make_client(response)
        result = get_chat_response(client, "question", "gpt-4")
        assert result["content"] == "the answer"

    def test_stop_reason_in_result(self):
        response = _make_response(finish_reason="length")
        client = _make_client(response)
        result = get_chat_response(client, "q", "gpt-4")
        assert result["stop_reason"] == "length"

    def test_reasoning_content_none_when_absent(self):
        response = _make_response(reasoning_content=None)
        client = _make_client(response)
        result = get_chat_response(client, "q", "gpt-4")
        assert result["reasoning_content"] is None

    def test_reasoning_content_returned_when_present(self):
        message = MagicMock()
        message.content = "answer"
        message.reasoning_content = "<think>step</think>"
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens = 1
        usage.completion_tokens = 1
        usage.total_tokens = 2
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        response.model_dump.return_value = {}
        client = _make_client(response)
        result = get_chat_response(client, "q", "gpt-4")
        assert result["reasoning_content"] == "<think>step</think>"

    def test_calls_api_with_model_and_messages(self):
        client = _make_client()
        get_chat_response(client, "my prompt", "gpt-3.5-turbo")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-3.5-turbo"
        assert kwargs["messages"] == [{"role": "user", "content": "my prompt"}]

    def test_temperature_included_when_set(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", temperature=0.7)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.7)

    def test_temperature_excluded_when_none(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", temperature=None)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in kwargs

    def test_max_tokens_included_when_set(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", max_tokens=512)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 512

    def test_max_tokens_excluded_when_none(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", max_tokens=None)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in kwargs

    def test_top_p_included_when_set(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", top_p=0.9)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["top_p"] == pytest.approx(0.9)

    def test_top_p_excluded_when_none(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", top_p=None)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "top_p" not in kwargs

    def test_frequency_penalty_included_when_set(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", frequency_penalty=0.5)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["frequency_penalty"] == pytest.approx(0.5)

    def test_presence_penalty_included_when_set(self):
        client = _make_client()
        get_chat_response(client, "q", "gpt-4", presence_penalty=0.3)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["presence_penalty"] == pytest.approx(0.3)

    def test_no_raw_key_by_default(self):
        client = _make_client()
        result = get_chat_response(client, "q", "gpt-4", return_raw=False)
        assert "raw" not in result

    def test_raw_key_present_when_return_raw_true(self):
        client = _make_client()
        result = get_chat_response(client, "q", "gpt-4", return_raw=True)
        assert "raw" in result

    def test_raw_contains_request_and_response(self):
        client = _make_client()
        result = get_chat_response(client, "q", "gpt-4", return_raw=True)
        assert "request" in result["raw"]
        assert "response" in result["raw"]

    def test_raw_request_has_model(self):
        client = _make_client()
        result = get_chat_response(client, "q", "my-model", return_raw=True)
        assert result["raw"]["request"]["model"] == "my-model"

    def test_authentication_error_reraises(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = AuthenticationError(
            message="Unauthorized", response=MagicMock(), body={}
        )
        with pytest.raises(AuthenticationError):
            get_chat_response(client, "q", "gpt-4")

    def test_none_content_logs_warning(self, caplog):
        response = _make_response(content=None)
        client = _make_client(response)
        with caplog.at_level(logging.WARNING, logger="syntk.api_client"):
            result = get_chat_response(client, "q", "gpt-4")
        assert result["content"] is None
        assert any("None" in r.message or "none" in r.message.lower() for r in caplog.records)
