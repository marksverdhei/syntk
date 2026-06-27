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


# ---------------------------------------------------------------------------
# save_dataframe — hf:// dataset repo auto-create
# ---------------------------------------------------------------------------

class TestSaveDataframeHfAutoCreate:
    def _install_fake_hf_hub(self, monkeypatch, captured):
        """Inject a fake `huggingface_hub` into sys.modules whose create_repo
        records its call args into `captured` — covers io._ensure_hf_dataset_repo_exists's
        lazy `from huggingface_hub import create_repo`."""
        import sys
        import types
        fake = types.ModuleType("huggingface_hub")
        def fake_create_repo(repo_id, repo_type, private, exist_ok):
            captured.append({"repo_id": repo_id, "repo_type": repo_type,
                             "private": private, "exist_ok": exist_ok})
        fake.create_repo = fake_create_repo
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    def test_extracts_owner_repo_from_hf_path(self, monkeypatch):
        from syntk.io import _ensure_hf_dataset_repo_exists
        captured = []
        self._install_fake_hf_hub(monkeypatch, captured)
        _ensure_hf_dataset_repo_exists("hf://datasets/myowner/myrepo/data/out.tsv")
        assert len(captured) == 1
        assert captured[0]["repo_id"] == "myowner/myrepo"

    def test_uses_dataset_repo_type_private_existok(self, monkeypatch):
        from syntk.io import _ensure_hf_dataset_repo_exists
        captured = []
        self._install_fake_hf_hub(monkeypatch, captured)
        _ensure_hf_dataset_repo_exists("hf://datasets/owner/repo/file.tsv")
        assert captured[0]["repo_type"] == "dataset"
        assert captured[0]["private"] is True
        assert captured[0]["exist_ok"] is True

    def test_no_call_when_path_lacks_owner_or_repo(self, monkeypatch):
        from syntk.io import _ensure_hf_dataset_repo_exists
        captured = []
        self._install_fake_hf_hub(monkeypatch, captured)
        _ensure_hf_dataset_repo_exists("hf://datasets/onlyowner")  # missing /repo
        assert captured == []

    def test_silently_swallows_create_repo_errors(self, monkeypatch):
        """If the network/token is missing, _ensure_hf_dataset_repo_exists
        should not raise — the subsequent write will surface a more useful error."""
        import sys
        import types
        fake = types.ModuleType("huggingface_hub")
        def boom(**kwargs):
            raise RuntimeError("simulated auth failure")
        fake.create_repo = boom
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
        from syntk.io import _ensure_hf_dataset_repo_exists
        # Must not raise
        _ensure_hf_dataset_repo_exists("hf://datasets/owner/repo/x.csv")

    def test_save_dataframe_routes_hf_paths_through_ensure(self, monkeypatch, tmp_path):
        """End-to-end-ish: save_dataframe with an hf:// path should call
        _ensure_hf_dataset_repo_exists, then route the actual write through
        pandas as usual (which we stub here to avoid touching the network)."""
        from syntk import io
        captured = []
        self._install_fake_hf_hub(monkeypatch, captured)
        # Replace to_csv so the test doesn't actually try to hit hf://
        write_calls = []
        monkeypatch.setattr(
            pd.DataFrame, "to_csv",
            lambda self, *a, **kw: write_calls.append((a, kw)),
        )
        df = pd.DataFrame({"x": [1, 2]})
        io.save_dataframe(df, "hf://datasets/own/rep/data/out.csv")
        assert len(captured) == 1
        assert captured[0]["repo_id"] == "own/rep"
        assert len(write_calls) == 1

    def test_save_dataframe_does_not_call_ensure_for_local_paths(self, monkeypatch, tmp_path):
        from syntk import io
        captured = []
        self._install_fake_hf_hub(monkeypatch, captured)
        io.save_dataframe(pd.DataFrame({"x": [1]}), str(tmp_path / "out.csv"))
        assert captured == []
