"""Tests for syntk.pipelines.column — ColumnDataArguments, ColumnProcessingArguments,
and ColumnPipeline.get_argument_classes."""

from __future__ import annotations

import pytest

from syntk.pipelines.column import (
    ColumnDataArguments,
    ColumnPipeline,
    ColumnProcessingArguments,
)
from syntk.config import (
    APIArguments,
    ConfigArguments,
    GenerationArguments,
    BaseDataArguments,
    BaseProcessingArguments,
)
from syntk.tracking import TrackingArguments


# ---------------------------------------------------------------------------
# ColumnDataArguments defaults
# ---------------------------------------------------------------------------

class TestColumnDataArgumentsDefaults:
    def test_default_text_column(self):
        args = ColumnDataArguments()
        assert args.text_column == "text"

    def test_default_output_column(self):
        args = ColumnDataArguments()
        assert args.output_column == "generated"

    def test_default_reasoning_content_column_is_none(self):
        args = ColumnDataArguments()
        assert args.reasoning_content_column is None

    def test_default_save_stop_reason_is_false(self):
        args = ColumnDataArguments()
        assert args.save_stop_reason is False

    def test_default_raw_api_json_path_is_none(self):
        args = ColumnDataArguments()
        assert args.raw_api_json_path is None

    def test_inherits_base_data_defaults(self):
        args = ColumnDataArguments()
        assert args.input_file == "input.parquet"
        assert args.output_file == "output.parquet"
        assert args.limit is None

    def test_custom_text_column(self):
        args = ColumnDataArguments(text_column="content")
        assert args.text_column == "content"

    def test_custom_output_column(self):
        args = ColumnDataArguments(output_column="result")
        assert args.output_column == "result"

    def test_save_stop_reason_true(self):
        args = ColumnDataArguments(save_stop_reason=True)
        assert args.save_stop_reason is True

    def test_reasoning_content_column_set(self):
        args = ColumnDataArguments(reasoning_content_column="reasoning")
        assert args.reasoning_content_column == "reasoning"


# ---------------------------------------------------------------------------
# ColumnProcessingArguments defaults
# ---------------------------------------------------------------------------

class TestColumnProcessingArgumentsDefaults:
    def test_default_prompt_template(self):
        args = ColumnProcessingArguments()
        assert "text" in args.prompt_template.lower()

    def test_prompt_template_contains_placeholder(self):
        args = ColumnProcessingArguments()
        assert "{text}" in args.prompt_template

    def test_inherits_base_processing_defaults(self):
        args = ColumnProcessingArguments()
        assert args.log_interval == 10
        assert args.save_interval == 100

    def test_custom_prompt_template(self):
        template = "Classify: {text}"
        args = ColumnProcessingArguments(prompt_template=template)
        assert args.prompt_template == "Classify: {text}"


# ---------------------------------------------------------------------------
# ColumnPipeline.get_argument_classes
# ---------------------------------------------------------------------------

class TestColumnPipelineArgumentClasses:
    def setup_method(self):
        self.pipeline = ColumnPipeline()

    def test_returns_tuple(self):
        classes = self.pipeline.get_argument_classes()
        assert isinstance(classes, tuple)

    def test_contains_config_arguments(self):
        classes = self.pipeline.get_argument_classes()
        assert ConfigArguments in classes

    def test_contains_api_arguments(self):
        classes = self.pipeline.get_argument_classes()
        assert APIArguments in classes

    def test_contains_generation_arguments(self):
        classes = self.pipeline.get_argument_classes()
        assert GenerationArguments in classes

    def test_contains_column_data_arguments(self):
        classes = self.pipeline.get_argument_classes()
        assert ColumnDataArguments in classes

    def test_contains_column_processing_arguments(self):
        classes = self.pipeline.get_argument_classes()
        assert ColumnProcessingArguments in classes

    def test_contains_tracking_arguments(self):
        classes = self.pipeline.get_argument_classes()
        assert TrackingArguments in classes

    def test_non_empty(self):
        classes = self.pipeline.get_argument_classes()
        assert len(classes) > 0
