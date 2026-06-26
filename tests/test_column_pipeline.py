"""Tests for ColumnPipeline.setup_dataframe and get_rows_to_process."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
)
from syntk.tracking import TrackingArguments
from syntk.pipelines.column import ColumnPipeline, ColumnDataArguments, ColumnProcessingArguments


def _make_pipeline(
    output_column: str = "generated",
    text_column: str = "text",
    save_stop_reason: bool = False,
    reasoning_content_column: Optional[str] = None,
    raw_api_json_path: Optional[str] = None,
) -> ColumnPipeline:
    pipeline = ColumnPipeline.__new__(ColumnPipeline)
    # Initialize base attributes without calling __init__ chain
    pipeline.responses = {}
    pipeline.client = None
    pipeline.tracker = MagicMock()
    pipeline.df = None
    pipeline.actual_stop_reason_column = None

    pipeline.config_args = ConfigArguments()
    pipeline.api_args = APIArguments()
    pipeline.gen_args = GenerationArguments()
    pipeline.data_args = ColumnDataArguments(
        output_column=output_column,
        text_column=text_column,
        save_stop_reason=save_stop_reason,
        reasoning_content_column=reasoning_content_column,
        raw_api_json_path=raw_api_json_path,
    )
    pipeline.proc_args = ColumnProcessingArguments()
    pipeline.track_args = TrackingArguments()
    return pipeline


# ---------------------------------------------------------------------------
# setup_dataframe
# ---------------------------------------------------------------------------

class TestSetupDataframe:
    def test_adds_output_column(self):
        pipeline = _make_pipeline(output_column="result")
        df = pd.DataFrame({"text": ["a", "b"]})
        result = pipeline.setup_dataframe(df, resuming=False)
        assert "result" in result.columns

    def test_output_column_initialized_to_na(self):
        pipeline = _make_pipeline(output_column="out")
        df = pd.DataFrame({"text": ["a"]})
        result = pipeline.setup_dataframe(df, resuming=False)
        assert pd.isna(result["out"].iloc[0])

    def test_returns_dataframe(self):
        pipeline = _make_pipeline()
        df = pd.DataFrame({"text": ["a"]})
        result = pipeline.setup_dataframe(df, resuming=False)
        assert isinstance(result, pd.DataFrame)

    def test_resume_keeps_existing_output_column(self):
        """When resuming and output column already exists, keep existing values."""
        pipeline = _make_pipeline(output_column="out")
        df = pd.DataFrame({"text": ["a", "b"], "out": ["done", pd.NA]})
        result = pipeline.setup_dataframe(df, resuming=True)
        assert result["out"].iloc[0] == "done"

    def test_adds_stop_reason_column_when_flag_set(self):
        pipeline = _make_pipeline(save_stop_reason=True)
        df = pd.DataFrame({"text": ["a"]})
        result = pipeline.setup_dataframe(df, resuming=False)
        assert pipeline.actual_stop_reason_column is not None
        assert pipeline.actual_stop_reason_column in result.columns

    def test_no_stop_reason_column_when_flag_not_set(self):
        pipeline = _make_pipeline(save_stop_reason=False)
        df = pd.DataFrame({"text": ["a"]})
        pipeline.setup_dataframe(df, resuming=False)
        assert pipeline.actual_stop_reason_column is None

    def test_resume_reuses_pipeline_stop_reason_column(self):
        # On resume, reuse the stop_reason column written last run instead of
        # allocating a new suffix that splits the data across runs (#44).
        pipeline = _make_pipeline(output_column="out", save_stop_reason=True)
        # Run 1's partial output: row 0 processed (out + stop_reason both set).
        df = pd.DataFrame({
            "text": ["a", "b"],
            "out": ["gen-0", pd.NA],
            "stop_reason": ["stop", pd.NA],
        })
        pipeline.setup_dataframe(df, resuming=True)
        assert pipeline.actual_stop_reason_column == "stop_reason"

    def test_resume_reuses_owned_column_despite_user_collision(self):
        # If the input had a user 'stop_reason' column, Run 1 wrote to
        # 'stop_reason_1'; resume must reuse that, not allocate 'stop_reason_2'.
        pipeline = _make_pipeline(output_column="out", save_stop_reason=True)
        df = pd.DataFrame({
            "text": ["a", "b"],
            "stop_reason": ["user-x", "user-y"],   # user's full column
            "out": ["gen-0", pd.NA],
            "stop_reason_1": ["stop", pd.NA],       # pipeline-owned, partial
        })
        pipeline.setup_dataframe(df, resuming=True)
        assert pipeline.actual_stop_reason_column == "stop_reason_1"

    def test_adds_reasoning_content_column_when_specified(self):
        pipeline = _make_pipeline(reasoning_content_column="reasoning")
        df = pd.DataFrame({"text": ["a"]})
        result = pipeline.setup_dataframe(df, resuming=False)
        assert "reasoning" in result.columns

    def test_creates_raw_api_json_file(self, tmp_path):
        raw_path = str(tmp_path / "raw.jsonl")
        pipeline = _make_pipeline(raw_api_json_path=raw_path)
        df = pd.DataFrame({"text": ["a"]})
        pipeline.setup_dataframe(df, resuming=False)
        assert Path(raw_path).exists()

    def test_no_raw_file_when_path_not_set(self, tmp_path):
        pipeline = _make_pipeline(raw_api_json_path=None)
        df = pd.DataFrame({"text": ["a"]})
        pipeline.setup_dataframe(df, resuming=False)
        # Nothing created in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_creates_nested_raw_api_directory(self, tmp_path):
        raw_path = str(tmp_path / "subdir" / "raw.jsonl")
        pipeline = _make_pipeline(raw_api_json_path=raw_path)
        df = pd.DataFrame({"text": ["a"]})
        pipeline.setup_dataframe(df, resuming=False)
        assert Path(raw_path).exists()


# ---------------------------------------------------------------------------
# get_rows_to_process
# ---------------------------------------------------------------------------

class TestGetRowsToProcess:
    def test_returns_all_rows_when_column_missing(self):
        pipeline = _make_pipeline(output_column="out")
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        rows = pipeline.get_rows_to_process(df)
        assert rows == [0, 1, 2]

    def test_returns_only_na_rows(self):
        pipeline = _make_pipeline(output_column="out")
        df = pd.DataFrame({"text": ["a", "b", "c"], "out": ["done", pd.NA, "done"]})
        rows = pipeline.get_rows_to_process(df)
        assert rows == [1]

    def test_returns_empty_when_all_done(self):
        pipeline = _make_pipeline(output_column="out")
        df = pd.DataFrame({"text": ["a", "b"], "out": ["x", "y"]})
        rows = pipeline.get_rows_to_process(df)
        assert rows == []

    def test_returns_all_when_all_na(self):
        pipeline = _make_pipeline(output_column="out")
        df = pd.DataFrame({"text": ["a", "b", "c"], "out": [pd.NA, pd.NA, pd.NA]})
        rows = pipeline.get_rows_to_process(df)
        assert len(rows) == 3

    def test_returns_list_type(self):
        pipeline = _make_pipeline(output_column="out")
        df = pd.DataFrame({"text": ["a"]})
        rows = pipeline.get_rows_to_process(df)
        assert isinstance(rows, list)
