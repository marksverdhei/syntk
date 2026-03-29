"""Tests for syntk.pipelines.base — BasePipeline.apply_limit and initialize_client."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Type
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from syntk.pipelines.base import BasePipeline
from syntk.config import (
    APIArguments,
    BaseDataArguments,
    BaseProcessingArguments,
    GenerationArguments,
    ConfigArguments,
)
from syntk.tracking import TrackingArguments


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------

class _MinimalPipeline(BasePipeline):
    def get_argument_classes(self) -> Tuple[Type, ...]:
        return ()

    def setup_dataframe(self, df, resuming):
        return df

    def process_row(self, row, idx):
        return {}

    def get_config_params(self):
        return {}

    def get_rows_to_process(self, df):
        return list(df.index)


def _make_pipeline(limit=None) -> _MinimalPipeline:
    p = _MinimalPipeline()
    p.data_args = BaseDataArguments(
        input_file="dummy.csv",
        output_file="dummy_out.csv",
        limit=limit,
    )
    p.api_args = APIArguments()
    p.gen_args = GenerationArguments()
    p.proc_args = BaseProcessingArguments()
    p.track_args = TrackingArguments()
    p.config_args = ConfigArguments()
    return p


def _df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"x": list(range(n))})


# ---------------------------------------------------------------------------
# apply_limit
# ---------------------------------------------------------------------------

class TestApplyLimit:
    def test_none_limit_returns_full_df(self):
        p = _make_pipeline(limit=None)
        df = _df(10)
        result = p.apply_limit(df)
        assert len(result) == 10

    def test_integer_limit_truncates(self):
        p = _make_pipeline(limit=5)
        result = p.apply_limit(_df(10))
        assert len(result) == 5

    def test_integer_limit_larger_than_df(self):
        """Limit larger than df → returns all rows."""
        p = _make_pipeline(limit=100)
        result = p.apply_limit(_df(10))
        assert len(result) == 10

    def test_fraction_limit_truncates(self):
        """0 < limit < 1 is treated as a fraction."""
        p = _make_pipeline(limit=0.5)
        result = p.apply_limit(_df(10))
        assert len(result) == 5

    def test_fraction_limit_small(self):
        p = _make_pipeline(limit=0.1)
        result = p.apply_limit(_df(100))
        assert len(result) == 10

    def test_zero_limit_raises(self):
        p = _make_pipeline(limit=0)
        with pytest.raises(ValueError, match="[Ll]imit"):
            p.apply_limit(_df(10))

    def test_negative_limit_raises(self):
        p = _make_pipeline(limit=-1)
        with pytest.raises(ValueError, match="[Ll]imit"):
            p.apply_limit(_df(10))

    def test_limit_one_keeps_one_row(self):
        p = _make_pipeline(limit=1)
        result = p.apply_limit(_df(10))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# initialize_client
# ---------------------------------------------------------------------------

class TestInitializeClient:
    def test_uses_env_api_key(self, monkeypatch):
        p = _make_pipeline()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-abc")
        with patch("syntk.pipelines.base.OpenAI") as mock_openai:
            p.initialize_client()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key-abc"

    def test_uses_base_url_from_api_args(self, monkeypatch):
        p = _make_pipeline()
        p.api_args.base_url = "http://myserver:8080/v1"
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        with patch("syntk.pipelines.base.OpenAI") as mock_openai:
            p.initialize_client()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "http://myserver:8080/v1"

    def test_placeholder_key_when_env_missing(self, monkeypatch):
        p = _make_pipeline()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("syntk.pipelines.base.OpenAI") as mock_openai:
            p.initialize_client()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "placeholder-api-key"

    def test_client_set_on_instance(self, monkeypatch):
        p = _make_pipeline()
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        fake_client = MagicMock()
        with patch("syntk.pipelines.base.OpenAI", return_value=fake_client):
            p.initialize_client()
        assert p.client is fake_client

    def test_custom_api_key_env_var(self, monkeypatch):
        p = _make_pipeline()
        p.api_args.api_key_env = "MY_CUSTOM_KEY"
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-secret")
        with patch("syntk.pipelines.base.OpenAI") as mock_openai:
            p.initialize_client()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "custom-secret"
