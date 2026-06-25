"""Tests for syntk.config — configuration dataclass defaults and construction."""

from __future__ import annotations

import pytest

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
    BaseDataArguments,
    BaseProcessingArguments,
)


# ---------------------------------------------------------------------------
# ConfigArguments
# ---------------------------------------------------------------------------

class TestConfigArguments:
    def test_default_config_file_is_none(self):
        args = ConfigArguments()
        assert args.config_file is None

    def test_custom_config_file(self):
        args = ConfigArguments(config_file="config.yaml")
        assert args.config_file == "config.yaml"


# ---------------------------------------------------------------------------
# APIArguments
# ---------------------------------------------------------------------------

class TestAPIArguments:
    def test_default_base_url(self):
        args = APIArguments()
        assert args.base_url == "http://localhost:8000/v1"

    def test_default_api_key_env(self):
        args = APIArguments()
        assert args.api_key_env == "OPENAI_API_KEY"

    def test_default_model(self):
        args = APIArguments()
        assert args.model == "gpt-3.5-turbo"

    def test_custom_values(self):
        args = APIArguments(
            base_url="https://api.openai.com/v1",
            api_key_env="MY_KEY",
            model="gpt-4o",
        )
        assert args.base_url == "https://api.openai.com/v1"
        assert args.api_key_env == "MY_KEY"
        assert args.model == "gpt-4o"


# ---------------------------------------------------------------------------
# GenerationArguments
# ---------------------------------------------------------------------------

class TestGenerationArguments:
    def test_all_defaults_none(self):
        args = GenerationArguments()
        assert args.temperature is None
        assert args.max_tokens is None
        assert args.top_p is None
        assert args.frequency_penalty is None
        assert args.presence_penalty is None

    def test_temperature_set(self):
        args = GenerationArguments(temperature=0.7)
        assert args.temperature == 0.7

    def test_max_tokens_set(self):
        args = GenerationArguments(max_tokens=1024)
        assert args.max_tokens == 1024

    def test_all_set(self):
        args = GenerationArguments(
            temperature=0.5,
            max_tokens=512,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )
        assert args.temperature == 0.5
        assert args.max_tokens == 512
        assert args.top_p == 0.9
        assert args.frequency_penalty == 0.1
        assert args.presence_penalty == 0.2


# ---------------------------------------------------------------------------
# BaseDataArguments
# ---------------------------------------------------------------------------

class TestBaseDataArguments:
    def test_default_input_file(self):
        args = BaseDataArguments()
        assert args.input_file == "input.parquet"

    def test_default_output_file(self):
        args = BaseDataArguments()
        assert args.output_file == "output.parquet"

    def test_default_limit_is_none(self):
        args = BaseDataArguments()
        assert args.limit is None

    def test_custom_files(self):
        args = BaseDataArguments(input_file="data.csv", output_file="result.jsonl")
        assert args.input_file == "data.csv"
        assert args.output_file == "result.jsonl"

    def test_integer_limit(self):
        args = BaseDataArguments(limit=100)
        assert args.limit == 100

    def test_fraction_limit(self):
        args = BaseDataArguments(limit=0.5)
        assert args.limit == 0.5


# ---------------------------------------------------------------------------
# BaseProcessingArguments
# ---------------------------------------------------------------------------

class TestBaseProcessingArguments:
    def test_default_log_interval(self):
        args = BaseProcessingArguments()
        assert args.log_interval == 10

    def test_default_save_interval(self):
        args = BaseProcessingArguments()
        assert args.save_interval == 100

    def test_custom_intervals(self):
        args = BaseProcessingArguments(log_interval=0, save_interval=50)
        assert args.log_interval == 0
        assert args.save_interval == 50
