"""Tests for BasePipeline.load_data resume and fresh load logic."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
    BaseDataArguments,
    BaseProcessingArguments,
)
from syntk.tracking import TrackingArguments
from syntk.io import save_dataframe
from syntk.pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Minimal concrete pipeline for testing
# ---------------------------------------------------------------------------

class MinimalPipeline(BasePipeline):
    def __init__(self, input_file: str, output_file: str = "nonexistent_output.parquet"):
        super().__init__()
        self.config_args = ConfigArguments()
        self.api_args = APIArguments()
        self.gen_args = GenerationArguments()
        self.data_args = BaseDataArguments(
            input_file=input_file,
            output_file=output_file,
        )
        self.proc_args = BaseProcessingArguments()
        self.track_args = TrackingArguments()
        self._unprocessed_flag = True  # controls get_rows_to_process

    def get_argument_classes(self) -> Tuple[Type, ...]:
        return (
            ConfigArguments,
            APIArguments,
            GenerationArguments,
            BaseDataArguments,
            BaseProcessingArguments,
            TrackingArguments,
        )

    def setup_dataframe(self, df: pd.DataFrame, resuming: bool) -> pd.DataFrame:
        return df

    def process_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        return {}

    def get_config_params(self) -> Dict[str, Any]:
        return {}

    def get_rows_to_process(self, df: pd.DataFrame) -> List[int]:
        # When _unprocessed_flag is True, all rows need processing (nothing done yet).
        if self._unprocessed_flag:
            return list(range(len(df)))
        # When False, simulate that half the rows are already done.
        return list(range(len(df) // 2, len(df)))


def _write_csv(path: Path, rows: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"text": [f"row{i}" for i in range(rows)]})
    save_dataframe(df, str(path))
    return df


# ---------------------------------------------------------------------------
# Fresh load (no output file)
# ---------------------------------------------------------------------------

class TestLoadDataFresh:
    def test_loads_input_file_when_no_output(self, tmp_path):
        input_path = tmp_path / "input.csv"
        _write_csv(input_path)
        pipeline = MinimalPipeline(str(input_path), str(tmp_path / "output.parquet"))
        df, resuming = pipeline.load_data()
        assert len(df) == 5
        assert not resuming

    def test_resuming_flag_false_on_fresh_load(self, tmp_path):
        input_path = tmp_path / "input.csv"
        _write_csv(input_path)
        pipeline = MinimalPipeline(str(input_path))
        _, resuming = pipeline.load_data()
        assert resuming is False

    def test_returns_dataframe_type(self, tmp_path):
        input_path = tmp_path / "input.csv"
        _write_csv(input_path)
        pipeline = MinimalPipeline(str(input_path))
        df, _ = pipeline.load_data()
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Resume from output file (partial processing)
# ---------------------------------------------------------------------------

class TestLoadDataResume:
    def test_resumes_when_output_file_exists_and_partially_processed(self, tmp_path):
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        _write_csv(input_path, rows=10)
        # Write the same data to the output file — simulates an existing checkpoint
        _write_csv(output_path, rows=10)

        pipeline = MinimalPipeline(str(input_path), str(output_path))
        pipeline._unprocessed_flag = False  # some rows done

        df, resuming = pipeline.load_data()
        assert resuming is True
        assert len(df) == 10

    def test_no_resume_when_all_rows_unprocessed(self, tmp_path):
        """If output file exists but no rows are processed, load input instead."""
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        _write_csv(input_path, rows=4)
        _write_csv(output_path, rows=4)

        pipeline = MinimalPipeline(str(input_path), str(output_path))
        pipeline._unprocessed_flag = True  # all rows still need processing

        df, resuming = pipeline.load_data()
        assert not resuming

    def test_falls_back_to_input_on_corrupt_output(self, tmp_path):
        """Corrupt output file → falls back to input file without raising."""
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        _write_csv(input_path, rows=3)
        output_path.write_text("not a valid csv file !!!")

        pipeline = MinimalPipeline(str(input_path), str(output_path))
        df, resuming = pipeline.load_data()
        assert not resuming
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Row count preservation
# ---------------------------------------------------------------------------

class TestLoadDataRowCount:
    def test_row_count_matches_input(self, tmp_path):
        input_path = tmp_path / "input.csv"
        original = pd.DataFrame({"a": range(7)})
        save_dataframe(original, str(input_path))

        pipeline = MinimalPipeline(str(input_path))
        df, _ = pipeline.load_data()
        assert len(df) == 7

    def test_columns_preserved(self, tmp_path):
        input_path = tmp_path / "input.csv"
        original = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        save_dataframe(original, str(input_path))

        pipeline = MinimalPipeline(str(input_path))
        df, _ = pipeline.load_data()
        assert list(df.columns) == ["x", "y"]
