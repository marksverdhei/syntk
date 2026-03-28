"""Tests for syntk.io — save_dataframe / load_dataframe / find_available_column_name."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from syntk.io import save_dataframe, load_dataframe, find_available_column_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# ---------------------------------------------------------------------------
# save_dataframe — local paths
# ---------------------------------------------------------------------------

class TestSaveDataframeLocal:
    def test_csv(self, tmp_path):
        df = _simple_df()
        out = tmp_path / "out.csv"
        save_dataframe(df, str(out))
        assert out.exists()
        loaded = pd.read_csv(out)
        assert list(loaded.columns) == ["a", "b"]

    def test_tsv(self, tmp_path):
        df = _simple_df()
        out = tmp_path / "out.tsv"
        save_dataframe(df, str(out))
        loaded = pd.read_csv(out, sep="\t")
        assert list(loaded.columns) == ["a", "b"]

    def test_json(self, tmp_path):
        df = _simple_df()
        out = tmp_path / "out.json"
        save_dataframe(df, str(out))
        loaded = pd.read_json(out)
        assert len(loaded) == 3

    def test_jsonl(self, tmp_path):
        df = _simple_df()
        out = tmp_path / "out.jsonl"
        save_dataframe(df, str(out))
        loaded = pd.read_json(out, lines=True)
        assert len(loaded) == 3

    def test_parquet(self, tmp_path):
        df = _simple_df()
        out = tmp_path / "out.parquet"
        save_dataframe(df, str(out))
        loaded = pd.read_parquet(out)
        assert list(loaded.columns) == ["a", "b"]

    def test_creates_nested_directories(self, tmp_path):
        df = _simple_df()
        out = tmp_path / "a" / "b" / "c" / "out.csv"
        save_dataframe(df, str(out))
        assert out.exists()

    def test_unsupported_format_raises(self, tmp_path):
        df = _simple_df()
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_dataframe(df, str(tmp_path / "out.xlsx"))

    def test_uppercase_extension(self, tmp_path):
        df = _simple_df()
        out = tmp_path / "OUT.CSV"
        save_dataframe(df, str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# save_dataframe — hf:// paths
# ---------------------------------------------------------------------------

class TestSaveDataframeHf:
    def test_calls_create_repo_with_correct_repo_id(self):
        df = _simple_df()
        hf_path = "hf://datasets/myowner/myrepo/data/out.tsv"
        mock_create_repo = MagicMock()
        mock_to_csv = MagicMock()

        with patch("huggingface_hub.create_repo", mock_create_repo), \
             patch.object(pd.DataFrame, "to_csv", mock_to_csv):
            with patch("syntk.io.create_repo", mock_create_repo, create=True):
                pass  # just ensure import path

        # Patch at the import location inside the function
        import importlib
        import syntk.io as io_module

        captured_calls = []

        def fake_create_repo(repo_id, repo_type, private, exist_ok):
            captured_calls.append((repo_id, repo_type, private, exist_ok))

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: (
            type("M", (), {"create_repo": fake_create_repo})()
            if name == "huggingface_hub" else original_import(name, *args, **kwargs)
        )):
            with patch.object(pd.DataFrame, "to_csv"):
                try:
                    save_dataframe(df, hf_path)
                except Exception:
                    pass

        # The real test: verify repo_id extraction logic
        from syntk.io import _ensure_hf_dataset_repo_exists
        captured = []

        def fake_create(repo_id, repo_type, private, exist_ok):
            captured.append({"repo_id": repo_id, "repo_type": repo_type,
                              "private": private, "exist_ok": exist_ok})

        with patch("huggingface_hub.create_repo", fake_create):
            import sys
            # Ensure create_repo is mockable via the lazy import in the function
            hf_hub_mock = MagicMock()
            hf_hub_mock.create_repo = fake_create
            sys.modules["huggingface_hub"] = hf_hub_mock
            try:
                _ensure_hf_dataset_repo_exists("hf://datasets/myowner/myrepo/data/out.tsv")
            finally:
                del sys.modules["huggingface_hub"]

        assert len(captured) == 1
        assert captured[0]["repo_id"] == "myowner/myrepo"
        assert captured[0]["repo_type"] == "dataset"
        assert captured[0]["private"] is True
        assert captured[0]["exist_ok"] is True

    def test_create_repo_called_with_exist_ok(self):
        from syntk.io import _ensure_hf_dataset_repo_exists
        import sys

        calls = []
        hf_hub_mock = MagicMock()
        hf_hub_mock.create_repo = lambda **kw: calls.append(kw) or None

        def fake_create_repo(repo_id, repo_type, private, exist_ok):
            calls.append(dict(repo_id=repo_id, repo_type=repo_type,
                              private=private, exist_ok=exist_ok))

        hf_hub_mock.create_repo = fake_create_repo
        sys.modules["huggingface_hub"] = hf_hub_mock
        try:
            _ensure_hf_dataset_repo_exists("hf://datasets/owner/repo/file.tsv")
        finally:
            del sys.modules["huggingface_hub"]

        assert calls[0]["exist_ok"] is True

    def test_short_hf_path_does_not_crash(self):
        from syntk.io import _ensure_hf_dataset_repo_exists
        # Should silently return without error
        _ensure_hf_dataset_repo_exists("hf://datasets/onlyone")

    def test_create_repo_error_is_swallowed(self):
        from syntk.io import _ensure_hf_dataset_repo_exists
        import sys

        hf_hub_mock = MagicMock()
        hf_hub_mock.create_repo = MagicMock(side_effect=RuntimeError("network error"))
        sys.modules["huggingface_hub"] = hf_hub_mock
        try:
            # Should not raise
            _ensure_hf_dataset_repo_exists("hf://datasets/owner/repo/file.tsv")
        finally:
            del sys.modules["huggingface_hub"]

    def test_hf_path_does_not_call_makedirs(self, tmp_path):
        from syntk.io import _ensure_hf_dataset_repo_exists
        import sys

        hf_hub_mock = MagicMock()
        hf_hub_mock.create_repo = MagicMock()
        sys.modules["huggingface_hub"] = hf_hub_mock

        df = _simple_df()
        hf_path = "hf://datasets/owner/repo/out.tsv"

        makedirs_calls = []
        original_makedirs = os.makedirs

        with patch("os.makedirs", side_effect=lambda *a, **kw: makedirs_calls.append(a)):
            with patch.object(pd.DataFrame, "to_csv"):
                try:
                    save_dataframe(df, hf_path)
                except Exception:
                    pass

        assert not makedirs_calls, "os.makedirs should not be called for hf:// paths"

        del sys.modules["huggingface_hub"]


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------

class TestLoadDataframe:
    def test_csv(self, tmp_path):
        df = _simple_df()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        result = load_dataframe(str(p))
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_tsv(self, tmp_path):
        df = _simple_df()
        p = tmp_path / "data.tsv"
        df.to_csv(p, sep="\t", index=False)
        result = load_dataframe(str(p))
        assert list(result.columns) == ["a", "b"]

    def test_json(self, tmp_path):
        df = _simple_df()
        p = tmp_path / "data.json"
        df.to_json(p, orient="records")
        result = load_dataframe(str(p))
        assert len(result) == 3

    def test_jsonl(self, tmp_path):
        df = _simple_df()
        p = tmp_path / "data.jsonl"
        df.to_json(p, orient="records", lines=True)
        result = load_dataframe(str(p))
        assert len(result) == 3

    def test_parquet(self, tmp_path):
        df = _simple_df()
        p = tmp_path / "data.parquet"
        df.to_parquet(p, index=False)
        result = load_dataframe(str(p))
        assert list(result.columns) == ["a", "b"]

    def test_unsupported_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(str(tmp_path / "data.xlsx"))

    def test_uppercase_extension(self, tmp_path):
        df = _simple_df()
        p = tmp_path / "DATA.CSV"
        df.to_csv(p, index=False)
        result = load_dataframe(str(p))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# find_available_column_name
# ---------------------------------------------------------------------------

class TestFindAvailableColumnName:
    def test_returns_base_when_not_present(self):
        df = pd.DataFrame({"a": [1]})
        assert find_available_column_name(df, "b") == "b"

    def test_returns_suffixed_when_present(self):
        df = pd.DataFrame({"score": [1]})
        assert find_available_column_name(df, "score") == "score_1"

    def test_skips_occupied_suffixes(self):
        df = pd.DataFrame({"score": [1], "score_1": [2]})
        assert find_available_column_name(df, "score") == "score_2"

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        assert find_available_column_name(df, "col") == "col"

    def test_multiple_conflicts(self):
        df = pd.DataFrame({f"x_{i}" if i else "x": [i] for i in range(4)})
        result = find_available_column_name(df, "x")
        assert result == "x_4"
