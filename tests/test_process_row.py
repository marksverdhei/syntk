"""Tests for ColumnPipeline.process_row."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from syntk.pipelines.column import ColumnPipeline, ColumnDataArguments, ColumnProcessingArguments
from syntk.config import ConfigArguments, APIArguments, GenerationArguments
from syntk.tracking import TrackingArguments


def _make_api_result(content="hello", reasoning=None, stop_reason="stop", raw=None):
    return {
        "content": content,
        "reasoning_content": reasoning,
        "stop_reason": stop_reason,
        "raw": raw,
    }


def _make_pipeline(
    prompt_template: str = "{text}",
    output_column: str = "generated",
    reasoning_content_column=None,
    save_stop_reason: bool = False,
    raw_api_json_path=None,
    model: str = "gpt-3.5-turbo",
) -> ColumnPipeline:
    pipeline = ColumnPipeline.__new__(ColumnPipeline)
    pipeline.responses = {}
    pipeline.client = MagicMock()
    pipeline.tracker = MagicMock()
    pipeline.df = None

    pipeline.config_args = ConfigArguments()
    pipeline.api_args = APIArguments(model=model, base_url="http://localhost/v1")
    pipeline.gen_args = GenerationArguments()
    pipeline.data_args = ColumnDataArguments(
        output_column=output_column,
        reasoning_content_column=reasoning_content_column,
        save_stop_reason=save_stop_reason,
        raw_api_json_path=raw_api_json_path,
    )
    pipeline.proc_args = ColumnProcessingArguments(prompt_template=prompt_template)
    pipeline.track_args = TrackingArguments()

    # actual_stop_reason_column is set by setup_dataframe; simulate it
    if save_stop_reason:
        pipeline.actual_stop_reason_column = "stop_reason"
    else:
        pipeline.actual_stop_reason_column = None

    return pipeline


class TestProcessRow:
    def _row(self, **kwargs):
        return pd.Series(kwargs)

    def test_returns_output_column(self):
        pipeline = _make_pipeline()
        row = self._row(text="hello world")
        api_result = _make_api_result(content="a response")
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            result = pipeline.process_row(row, idx=0)
        assert result["generated"] == "a response"

    def test_none_content_coerced_to_empty_string(self):
        # None content must be written as "" not None, else the row looks
        # unprocessed (NaN) and is retried forever on resume (#45).
        pipeline = _make_pipeline()
        row = self._row(text="hi")
        api_result = _make_api_result(content=None)
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            result = pipeline.process_row(row, idx=0)
        assert result["generated"] == ""

    def test_none_result_row_not_reprocessed_on_resume(self):
        # End-to-end of the resume gate: a row whose generation yielded None
        # content is considered processed and not picked up again (#45).
        pipeline = _make_pipeline()
        api_result = _make_api_result(content=None)
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            res = pipeline.process_row(pd.Series({"text": "hi"}), idx=0)
        df = pd.DataFrame({"text": ["hi"]})
        df["generated"] = pd.NA
        for col, val in res.items():
            df.at[0, col] = val
        assert pipeline.get_rows_to_process(df) == []

    def test_formats_prompt_from_template(self):
        pipeline = _make_pipeline(prompt_template="Summarize: {text}")
        row = self._row(text="the quick fox")
        api_result = _make_api_result()
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result) as mock_api:
            pipeline.process_row(row, idx=0)
        mock_api.assert_called_once()
        assert mock_api.call_args.kwargs["prompt"] == "Summarize: the quick fox"

    def test_multi_column_template(self):
        pipeline = _make_pipeline(prompt_template="Title: {title}\nBody: {body}")
        row = self._row(title="Foo", body="bar baz")
        api_result = _make_api_result()
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result) as mock_api:
            pipeline.process_row(row, idx=0)
        assert mock_api.call_args.kwargs["prompt"] == "Title: Foo\nBody: bar baz"

    def test_missing_column_raises_value_error(self):
        pipeline = _make_pipeline(prompt_template="{missing_col}")
        row = self._row(text="irrelevant")
        with pytest.raises(ValueError, match="missing_col"):
            with patch("syntk.pipelines.column.get_chat_response"):
                pipeline.process_row(row, idx=0)

    def test_caches_response_on_first_call(self):
        pipeline = _make_pipeline()
        row = self._row(text="same prompt")
        api_result = _make_api_result(content="cached result")
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result) as mock_api:
            pipeline.process_row(row, idx=0)
            pipeline.process_row(row, idx=1)
        assert mock_api.call_count == 1

    def test_uses_cached_result_on_second_call(self):
        pipeline = _make_pipeline()
        row = self._row(text="same prompt")
        api_result = _make_api_result(content="cached value")
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            pipeline.process_row(row, idx=0)
            result = pipeline.process_row(row, idx=1)
        assert result["generated"] == "cached value"

    def test_different_prompts_call_api_twice(self):
        pipeline = _make_pipeline()
        row_a = self._row(text="prompt A")
        row_b = self._row(text="prompt B")
        api_result = _make_api_result()
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result) as mock_api:
            pipeline.process_row(row_a, idx=0)
            pipeline.process_row(row_b, idx=1)
        assert mock_api.call_count == 2

    def test_reasoning_content_column_included_when_set(self):
        pipeline = _make_pipeline(reasoning_content_column="reasoning")
        row = self._row(text="hello")
        api_result = _make_api_result(content="answer", reasoning="<think>step</think>")
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            result = pipeline.process_row(row, idx=0)
        assert result["reasoning"] == "<think>step</think>"

    def test_reasoning_column_absent_when_not_set(self):
        pipeline = _make_pipeline(reasoning_content_column=None)
        row = self._row(text="hello")
        api_result = _make_api_result()
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            result = pipeline.process_row(row, idx=0)
        assert "reasoning" not in result

    def test_stop_reason_column_included_when_enabled(self):
        pipeline = _make_pipeline(save_stop_reason=True)
        row = self._row(text="hello")
        api_result = _make_api_result(stop_reason="length")
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            result = pipeline.process_row(row, idx=0)
        assert result["stop_reason"] == "length"

    def test_stop_reason_absent_when_disabled(self):
        pipeline = _make_pipeline(save_stop_reason=False)
        row = self._row(text="hello")
        api_result = _make_api_result()
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result):
            result = pipeline.process_row(row, idx=0)
        assert "stop_reason" not in result

    def test_raw_api_json_saved_when_path_set(self, tmp_path):
        raw_path = str(tmp_path / "api.jsonl")
        pipeline = _make_pipeline(raw_api_json_path=raw_path)
        row = self._row(text="hello")
        api_result = _make_api_result(content="resp")
        with (
            patch("syntk.pipelines.column.get_chat_response", return_value=api_result),
            patch("syntk.pipelines.column.save_raw_api_call") as mock_save,
        ):
            pipeline.process_row(row, idx=3)
        mock_save.assert_called_once_with(raw_path, 3, api_result)

    def test_raw_api_json_not_saved_when_path_not_set(self):
        pipeline = _make_pipeline(raw_api_json_path=None)
        row = self._row(text="hello")
        api_result = _make_api_result()
        with (
            patch("syntk.pipelines.column.get_chat_response", return_value=api_result),
            patch("syntk.pipelines.column.save_raw_api_call") as mock_save,
        ):
            pipeline.process_row(row, idx=0)
        mock_save.assert_not_called()

    def test_api_called_with_correct_model(self):
        pipeline = _make_pipeline(model="gpt-4o")
        row = self._row(text="hi")
        api_result = _make_api_result()
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result) as mock_api:
            pipeline.process_row(row, idx=0)
        assert mock_api.call_args.kwargs["model"] == "gpt-4o"

    def test_return_raw_true_when_raw_path_set(self):
        pipeline = _make_pipeline(raw_api_json_path="/some/path.jsonl")
        row = self._row(text="hi")
        api_result = _make_api_result()
        with (
            patch("syntk.pipelines.column.get_chat_response", return_value=api_result) as mock_api,
            patch("syntk.pipelines.column.save_raw_api_call"),
        ):
            pipeline.process_row(row, idx=0)
        assert mock_api.call_args.kwargs["return_raw"] is True

    def test_return_raw_false_when_no_raw_path(self):
        pipeline = _make_pipeline(raw_api_json_path=None)
        row = self._row(text="hi")
        api_result = _make_api_result()
        with patch("syntk.pipelines.column.get_chat_response", return_value=api_result) as mock_api:
            pipeline.process_row(row, idx=0)
        assert mock_api.call_args.kwargs["return_raw"] is False
