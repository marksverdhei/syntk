"""Tests for BasePipeline._process_rows checkpointing and metrics logic."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
    BaseDataArguments,
    BaseProcessingArguments,
)
from syntk.tracking import TrackingArguments
from syntk.pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Minimal pipeline that writes a fixed value to a 'result' column
# ---------------------------------------------------------------------------

class TrackingPipeline(BasePipeline):
    """Concrete pipeline that records which rows were processed."""

    def __init__(
        self,
        output_file: str = "/tmp/test_output.csv",
        log_interval: int = 0,
        save_interval: int = 0,
    ):
        super().__init__()
        self.config_args = ConfigArguments()
        self.api_args = APIArguments()
        self.gen_args = GenerationArguments()
        self.data_args = BaseDataArguments(
            input_file="input.parquet",
            output_file=output_file,
        )
        self.proc_args = BaseProcessingArguments(
            log_interval=log_interval,
            save_interval=save_interval,
        )
        self.track_args = TrackingArguments()
        self.tracker = MagicMock()
        self.responses = {}
        self.processed_indices: List[int] = []

    def get_argument_classes(self) -> Tuple[Type, ...]:
        return (
            ConfigArguments, APIArguments, GenerationArguments,
            BaseDataArguments, BaseProcessingArguments, TrackingArguments,
        )

    def setup_dataframe(self, df: pd.DataFrame, resuming: bool) -> pd.DataFrame:
        df = df.copy()
        df["result"] = None
        return df

    def process_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        self.processed_indices.append(idx)
        return {"result": f"done_{idx}"}

    def get_config_params(self) -> Dict[str, Any]:
        return {}

    def get_rows_to_process(self, df: pd.DataFrame) -> List[int]:
        return list(range(len(df)))


def _make_pipeline_with_df(tmp_path: Path, n_rows: int = 5, **kwargs) -> TrackingPipeline:
    output_file = str(tmp_path / "output.csv")
    pipeline = TrackingPipeline(output_file=output_file, **kwargs)
    pipeline.df = pd.DataFrame({"text": [f"row{i}" for i in range(n_rows)], "result": [None] * n_rows})
    return pipeline


# ---------------------------------------------------------------------------
# Basic processing
# ---------------------------------------------------------------------------

class TestProcessRowsBasic:
    def test_all_rows_processed(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=4)
        pipeline._process_rows(list(range(4)), resuming=False)
        assert sorted(pipeline.processed_indices) == [0, 1, 2, 3]

    def test_subset_of_rows_processed(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=5)
        pipeline._process_rows([1, 3], resuming=False)
        assert sorted(pipeline.processed_indices) == [1, 3]

    def test_results_written_to_df(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=3)
        pipeline._process_rows(list(range(3)), resuming=False)
        assert pipeline.df.at[0, "result"] == "done_0"
        assert pipeline.df.at[2, "result"] == "done_2"

    def test_empty_rows_list_calls_final_summary(self, tmp_path):
        """Even with no rows to process, final summary and save are called."""
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=3)
        with (
            patch.object(pipeline, "_log_final_summary") as mock_summary,
            patch.object(pipeline, "_save_final_output") as mock_save,
        ):
            pipeline._process_rows([], resuming=False)
        mock_summary.assert_called_once()
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class TestProcessRowsCheckpointing:
    def test_checkpoint_called_at_interval(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=6, save_interval=2)
        with patch.object(pipeline, "_save_checkpoint") as mock_ckpt:
            pipeline._process_rows(list(range(6)), resuming=False)
        # Should be called at row 2 and row 4 (not at 6 since that's the final save)
        assert mock_ckpt.call_count == 3  # at 2, 4, 6

    def test_no_checkpoint_when_interval_zero(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=5, save_interval=0)
        with patch.object(pipeline, "_save_checkpoint") as mock_ckpt:
            pipeline._process_rows(list(range(5)), resuming=False)
        mock_ckpt.assert_not_called()

    def test_final_save_called_always(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=3, save_interval=0)
        with patch.object(pipeline, "_save_final_output") as mock_save:
            pipeline._process_rows(list(range(3)), resuming=False)
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

class TestProcessRowsMetrics:
    def test_log_metrics_called_at_interval(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=4, log_interval=2)
        with (
            patch.object(pipeline, "_log_metrics") as mock_metrics,
            patch.object(pipeline, "_save_final_output"),
        ):
            pipeline._process_rows(list(range(4)), resuming=False)
        # Called at row 2 and row 4
        assert mock_metrics.call_count == 2

    def test_no_metrics_when_interval_zero(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=5, log_interval=0)
        with (
            patch.object(pipeline, "_log_metrics") as mock_metrics,
            patch.object(pipeline, "_save_final_output"),
        ):
            pipeline._process_rows(list(range(5)), resuming=False)
        mock_metrics.assert_not_called()

    def test_final_summary_always_called(self, tmp_path):
        pipeline = _make_pipeline_with_df(tmp_path, n_rows=3, log_interval=0)
        with (
            patch.object(pipeline, "_log_final_summary") as mock_summary,
            patch.object(pipeline, "_save_final_output"),
        ):
            pipeline._process_rows(list(range(3)), resuming=False)
        mock_summary.assert_called_once()
