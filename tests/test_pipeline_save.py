"""Tests for BasePipeline._save_checkpoint and _save_final_output."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type
from unittest.mock import MagicMock, patch

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
from syntk.io import load_dataframe
from syntk.pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Minimal concrete pipeline for testing
# ---------------------------------------------------------------------------

class MinimalPipeline(BasePipeline):
    def __init__(self, output_file: str = "output.parquet"):
        super().__init__()
        self.config_args = ConfigArguments()
        self.api_args = APIArguments()
        self.gen_args = GenerationArguments()
        self.data_args = BaseDataArguments(
            input_file="input.parquet",
            output_file=output_file,
        )
        self.proc_args = BaseProcessingArguments()
        self.track_args = TrackingArguments()
        self.tracker = MagicMock()
        self.responses = {}

    def get_argument_classes(self) -> Tuple[Type, ...]:
        return (
            ConfigArguments, APIArguments, GenerationArguments,
            BaseDataArguments, BaseProcessingArguments, TrackingArguments,
        )

    def setup_dataframe(self, df: pd.DataFrame, resuming: bool) -> pd.DataFrame:
        return df

    def process_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        return {}

    def get_config_params(self) -> Dict[str, Any]:
        return {}

    def get_rows_to_process(self, df: pd.DataFrame) -> List[int]:
        return list(range(len(df)))


def _make_pipeline(tmp_path: Path) -> MinimalPipeline:
    output_file = str(tmp_path / "output.csv")
    pipeline = MinimalPipeline(output_file=output_file)
    pipeline.df = pd.DataFrame({"text": ["a", "b", "c"], "result": [1, 2, 3]})
    return pipeline


# ---------------------------------------------------------------------------
# _save_checkpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_creates_output_file(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline._save_checkpoint(processed_count=2, total_count=3)
        assert Path(pipeline.data_args.output_file).exists()

    def test_saved_content_matches_df(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline._save_checkpoint(processed_count=3, total_count=3)
        loaded = load_dataframe(pipeline.data_args.output_file)
        assert len(loaded) == 3
        assert list(loaded.columns) == ["text", "result"]

    def test_checkpoint_overwrites_previous(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        # First save with 5-row df
        pipeline.df = pd.DataFrame({"x": range(5)})
        pipeline._save_checkpoint(1, 5)
        # Update df and save again
        pipeline.df = pd.DataFrame({"x": range(3)})
        pipeline._save_checkpoint(3, 3)
        loaded = load_dataframe(pipeline.data_args.output_file)
        assert len(loaded) == 3

    def test_works_with_zero_count(self, tmp_path):
        """Checkpoint with 0 processed rows should still write the file."""
        pipeline = _make_pipeline(tmp_path)
        pipeline._save_checkpoint(processed_count=0, total_count=0)
        assert Path(pipeline.data_args.output_file).exists()


# ---------------------------------------------------------------------------
# _save_final_output
# ---------------------------------------------------------------------------

class TestSaveFinalOutput:
    def test_creates_output_file(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline._save_final_output()
        assert Path(pipeline.data_args.output_file).exists()

    def test_saved_content_matches_df(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline._save_final_output()
        loaded = load_dataframe(pipeline.data_args.output_file)
        assert len(loaded) == 3

    def test_saves_all_columns(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline._save_final_output()
        loaded = load_dataframe(pipeline.data_args.output_file)
        assert set(loaded.columns) == {"text", "result"}

    def test_uses_configured_output_path(self, tmp_path):
        """Output is written to the path in data_args.output_file."""
        custom_path = tmp_path / "custom_out.csv"
        pipeline = MinimalPipeline(output_file=str(custom_path))
        pipeline.df = pd.DataFrame({"v": [42]})
        pipeline.tracker = MagicMock()
        pipeline.responses = {}
        pipeline._save_final_output()
        assert custom_path.exists()
