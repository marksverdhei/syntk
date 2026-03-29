"""Tests for syntk.io — save_dataframe, load_dataframe, find_available_column_name."""

from __future__ import annotations

import os

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"id": list(range(rows)), "text": [f"row{i}" for i in range(rows)]})


# ---------------------------------------------------------------------------
# save_dataframe / load_dataframe — format roundtrips
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_csv_roundtrip(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "data.csv")
        df = _df()
        save_dataframe(df, path)
        loaded = load_dataframe(path)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_tsv_roundtrip(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "data.tsv")
        df = _df()
        save_dataframe(df, path)
        loaded = load_dataframe(path)
        assert len(loaded) == len(df)

    def test_json_roundtrip(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "data.json")
        df = _df()
        save_dataframe(df, path)
        loaded = load_dataframe(path)
        assert len(loaded) == len(df)

    def test_jsonl_roundtrip(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "data.jsonl")
        df = _df()
        save_dataframe(df, path)
        loaded = load_dataframe(path)
        assert len(loaded) == len(df)

    def test_parquet_roundtrip(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "data.parquet")
        df = _df()
        save_dataframe(df, path)
        loaded = load_dataframe(path)
        assert len(loaded) == len(df)
        assert list(loaded["id"]) == list(df["id"])

    def test_case_insensitive_extension(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "data.CSV")
        df = _df(2)
        save_dataframe(df, path)
        loaded = load_dataframe(path)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# save_dataframe — edge cases
# ---------------------------------------------------------------------------

class TestSaveDataframe:
    def test_creates_output_directory(self, tmp_path):
        from syntk.io import save_dataframe
        nested = str(tmp_path / "nested" / "dir" / "out.csv")
        save_dataframe(_df(), nested)
        assert os.path.exists(nested)

    def test_unsupported_format_raises_value_error(self, tmp_path):
        from syntk.io import save_dataframe
        with pytest.raises(ValueError, match="Unsupported"):
            save_dataframe(_df(), str(tmp_path / "data.xlsx"))

    def test_csv_no_index_column(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "out.csv")
        save_dataframe(_df(), path)
        loaded = load_dataframe(path)
        # No "Unnamed: 0" index column
        assert "Unnamed: 0" not in loaded.columns


# ---------------------------------------------------------------------------
# load_dataframe — edge cases
# ---------------------------------------------------------------------------

class TestLoadDataframe:
    def test_unsupported_format_raises_value_error(self, tmp_path):
        from syntk.io import load_dataframe
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataframe(str(tmp_path / "data.xlsx"))

    def test_preserves_column_names(self, tmp_path):
        from syntk.io import save_dataframe, load_dataframe
        path = str(tmp_path / "data.parquet")
        df = pd.DataFrame({"alpha": [1], "beta": [2], "gamma": [3]})
        save_dataframe(df, path)
        loaded = load_dataframe(path)
        assert set(loaded.columns) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# find_available_column_name
# ---------------------------------------------------------------------------

class TestFindAvailableColumnName:
    def test_returns_base_when_not_present(self):
        from syntk.io import find_available_column_name
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert find_available_column_name(df, "c") == "c"

    def test_returns_name_1_when_base_taken(self):
        from syntk.io import find_available_column_name
        df = pd.DataFrame({"output": [1]})
        assert find_available_column_name(df, "output") == "output_1"

    def test_increments_past_taken_suffixes(self):
        from syntk.io import find_available_column_name
        df = pd.DataFrame({"output": [1], "output_1": [2]})
        assert find_available_column_name(df, "output") == "output_2"

    def test_no_collision_on_empty_df(self):
        from syntk.io import find_available_column_name
        df = pd.DataFrame()
        assert find_available_column_name(df, "result") == "result"

    def test_large_suffix_skip(self):
        """Skips all taken suffixes up to the first available one."""
        from syntk.io import find_available_column_name
        cols = {"x": [1], "x_1": [2], "x_2": [3], "x_3": [4]}
        df = pd.DataFrame(cols)
        assert find_available_column_name(df, "x") == "x_4"
