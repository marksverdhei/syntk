"""Tests for syntk.config dataclasses and syntk.io.find_available_column_name."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from syntk.config import (
    APIArguments,
    BaseDataArguments,
    BaseProcessingArguments,
    ConfigArguments,
    GenerationArguments,
)
from syntk.io import find_available_column_name, load_dataframe, save_dataframe


# ---------------------------------------------------------------------------
# ConfigArguments
# ---------------------------------------------------------------------------

class TestConfigArguments:
    def test_default_config_file_is_none(self):
        args = ConfigArguments()
        assert args.config_file is None

    def test_custom_config_file(self):
        args = ConfigArguments(config_file="/path/to/config.yaml")
        assert args.config_file == "/path/to/config.yaml"


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

    def test_custom_base_url(self):
        args = APIArguments(base_url="http://remote:8080/v1")
        assert args.base_url == "http://remote:8080/v1"

    def test_custom_model(self):
        args = APIArguments(model="claude-3-haiku")
        assert args.model == "claude-3-haiku"


# ---------------------------------------------------------------------------
# GenerationArguments
# ---------------------------------------------------------------------------

class TestGenerationArguments:
    def test_all_defaults_are_none(self):
        args = GenerationArguments()
        assert args.temperature is None
        assert args.max_tokens is None
        assert args.top_p is None
        assert args.frequency_penalty is None
        assert args.presence_penalty is None

    def test_custom_temperature(self):
        args = GenerationArguments(temperature=0.7)
        assert args.temperature == pytest.approx(0.7)

    def test_custom_max_tokens(self):
        args = GenerationArguments(max_tokens=512)
        assert args.max_tokens == 512


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

    def test_custom_input_file(self):
        args = BaseDataArguments(input_file="data.csv")
        assert args.input_file == "data.csv"

    def test_fractional_limit(self):
        args = BaseDataArguments(limit=0.5)
        assert args.limit == pytest.approx(0.5)


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

    def test_custom_log_interval(self):
        args = BaseProcessingArguments(log_interval=50)
        assert args.log_interval == 50

    def test_custom_save_interval(self):
        args = BaseProcessingArguments(save_interval=0)
        assert args.save_interval == 0


# ---------------------------------------------------------------------------
# find_available_column_name
# ---------------------------------------------------------------------------

class TestFindAvailableColumnName:
    def _df(self, *columns) -> pd.DataFrame:
        return pd.DataFrame(columns=list(columns))

    def test_base_name_available(self):
        df = self._df("other")
        assert find_available_column_name(df, "result") == "result"

    def test_base_name_taken_returns_numbered(self):
        df = self._df("result")
        assert find_available_column_name(df, "result") == "result_1"

    def test_numbered_taken_increments(self):
        df = self._df("result", "result_1")
        assert find_available_column_name(df, "result") == "result_2"

    def test_gap_skipped(self):
        # result_1 is taken but result_2 is not - should return result_1 though
        # Wait, the algorithm uses counter starting at 1
        df = self._df("result", "result_1", "result_2")
        assert find_available_column_name(df, "result") == "result_3"

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        assert find_available_column_name(df, "col") == "col"

    def test_base_not_in_columns_returns_base(self):
        df = self._df("a", "b", "c")
        assert find_available_column_name(df, "d") == "d"


# ---------------------------------------------------------------------------
# save_dataframe / load_dataframe (syntk.io) — roundtrip spot checks
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def _df(self):
        return pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    def test_roundtrip_csv(self, tmp_path):
        path = str(tmp_path / "data.csv")
        save_dataframe(self._df(), path)
        loaded = load_dataframe(path)
        pd.testing.assert_frame_equal(self._df(), loaded)

    def test_roundtrip_jsonl(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        save_dataframe(self._df(), path)
        loaded = load_dataframe(path)
        pd.testing.assert_frame_equal(self._df(), loaded)

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "out.csv")
        save_dataframe(self._df(), path)
        loaded = load_dataframe(path)
        assert len(loaded) == 3

    def test_unsupported_save_format_raises(self, tmp_path):
        path = str(tmp_path / "data.xlsx")
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_dataframe(self._df(), path)

    def test_unsupported_load_format_raises(self, tmp_path):
        path = str(tmp_path / "data.xlsx")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(path)
