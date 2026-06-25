"""Tests for ColumnPipeline.get_config_params."""

from unittest.mock import MagicMock

import pytest

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
)
from syntk.tracking import TrackingArguments
from syntk.pipelines.column import ColumnPipeline, ColumnDataArguments, ColumnProcessingArguments


def _make_pipeline(
    model: str = "gpt-3.5-turbo",
    base_url: str = "http://localhost:8080/v1",
    temperature: float = None,
    max_tokens: int = None,
    text_column: str = "text",
    output_column: str = "generated",
    limit: float = None,
    log_interval: int = 10,
    save_interval: int = 100,
) -> ColumnPipeline:
    pipeline = ColumnPipeline.__new__(ColumnPipeline)
    pipeline.responses = {}
    pipeline.client = None
    pipeline.tracker = MagicMock()
    pipeline.df = None
    pipeline.actual_stop_reason_column = None

    pipeline.config_args = ConfigArguments()
    pipeline.api_args = APIArguments(model=model, base_url=base_url)
    pipeline.gen_args = GenerationArguments(temperature=temperature, max_tokens=max_tokens)
    pipeline.data_args = ColumnDataArguments(
        text_column=text_column,
        output_column=output_column,
        limit=limit,
    )
    pipeline.proc_args = ColumnProcessingArguments(
        log_interval=log_interval,
        save_interval=save_interval,
    )
    pipeline.track_args = TrackingArguments()
    return pipeline


class TestGetConfigParams:
    def test_returns_dict(self):
        pipeline = _make_pipeline()
        assert isinstance(pipeline.get_config_params(), dict)

    def test_contains_model(self):
        pipeline = _make_pipeline(model="my-model")
        params = pipeline.get_config_params()
        assert params["model"] == "my-model"

    def test_contains_base_url(self):
        pipeline = _make_pipeline(base_url="http://myserver/v1")
        params = pipeline.get_config_params()
        assert params["base_url"] == "http://myserver/v1"

    def test_contains_temperature(self):
        pipeline = _make_pipeline(temperature=0.7)
        params = pipeline.get_config_params()
        assert params["temperature"] == pytest.approx(0.7)

    def test_temperature_none_when_not_set(self):
        pipeline = _make_pipeline(temperature=None)
        params = pipeline.get_config_params()
        assert params["temperature"] is None

    def test_contains_max_tokens(self):
        pipeline = _make_pipeline(max_tokens=512)
        params = pipeline.get_config_params()
        assert params["max_tokens"] == 512

    def test_contains_text_column(self):
        pipeline = _make_pipeline(text_column="content")
        params = pipeline.get_config_params()
        assert params["text_column"] == "content"

    def test_contains_output_column(self):
        pipeline = _make_pipeline(output_column="result")
        params = pipeline.get_config_params()
        assert params["output_column"] == "result"

    def test_contains_limit(self):
        pipeline = _make_pipeline(limit=0.5)
        params = pipeline.get_config_params()
        assert params["limit"] == pytest.approx(0.5)

    def test_contains_log_interval(self):
        pipeline = _make_pipeline(log_interval=25)
        params = pipeline.get_config_params()
        assert params["log_interval"] == 25

    def test_contains_save_interval(self):
        pipeline = _make_pipeline(save_interval=50)
        params = pipeline.get_config_params()
        assert params["save_interval"] == 50

    def test_contains_input_and_output_file(self):
        pipeline = _make_pipeline()
        params = pipeline.get_config_params()
        assert "input_file" in params
        assert "output_file" in params
