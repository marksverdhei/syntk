"""Tests for Hugging Face integration."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from syntk.io import save_dataframe, _parse_hf_url, _ensure_hf_repo_exists


class TestParseHfUrl:
    """Tests for parsing Hugging Face URLs."""

    def test_parse_dataset_url(self):
        """Test parsing a dataset URL."""
        url = "hf://datasets/username/repo-name/file.tsv"
        repo_id, file_path = _parse_hf_url(url)
        assert repo_id == "username/repo-name"
        assert file_path == "file.tsv"

    def test_parse_dataset_url_with_nested_path(self):
        """Test parsing a dataset URL with nested file path."""
        url = "hf://datasets/username/repo-name/data/subfolder/file.parquet"
        repo_id, file_path = _parse_hf_url(url)
        assert repo_id == "username/repo-name"
        assert file_path == "data/subfolder/file.parquet"

    def test_parse_model_url(self):
        """Test parsing a model URL."""
        url = "hf://models/username/model-name/weights.bin"
        repo_id, file_path = _parse_hf_url(url)
        assert repo_id == "username/model-name"
        assert file_path == "weights.bin"

    def test_invalid_url_format(self):
        """Test that invalid URL format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Hugging Face URL format"):
            _parse_hf_url("hf://incomplete/url")


class TestEnsureHfRepoExists:
    """Tests for ensuring HF repository exists."""

    def test_skips_non_hf_urls(self):
        """Test that non-HF URLs are skipped."""
        with patch("syntk.io.HfApi") as mock_api:
            _ensure_hf_repo_exists("/local/path/file.csv")
            mock_api.assert_not_called()

    @patch("syntk.io.HfApi")
    def test_creates_private_dataset_repo(self, mock_hf_api):
        """Test that a private dataset repository is created."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance

        url = "hf://datasets/username/new-repo/file.tsv"
        _ensure_hf_repo_exists(url)

        mock_api_instance.create_repo.assert_called_once_with(
            repo_id="username/new-repo",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

    @patch("syntk.io.HfApi")
    def test_creates_model_repo(self, mock_hf_api):
        """Test that a model repository is created (repo_type=None for models)."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance

        url = "hf://models/username/new-model/weights.bin"
        _ensure_hf_repo_exists(url)

        mock_api_instance.create_repo.assert_called_once_with(
            repo_id="username/new-model", repo_type=None, private=True, exist_ok=True
        )

    @patch("syntk.io.HfApi")
    def test_handles_existing_repo(self, mock_hf_api):
        """Test that existing repositories are handled gracefully."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        # exist_ok=True should handle this case

        url = "hf://datasets/username/existing-repo/file.tsv"
        _ensure_hf_repo_exists(url)

        mock_api_instance.create_repo.assert_called_once()

    @patch("syntk.io.HfApi")
    def test_handles_api_errors_gracefully(self, mock_hf_api):
        """Test that API errors are handled without raising exceptions."""
        mock_api_instance = Mock()
        mock_api_instance.create_repo.side_effect = Exception("API error")
        mock_hf_api.return_value = mock_api_instance

        url = "hf://datasets/username/new-repo/file.tsv"
        # Should not raise an exception
        _ensure_hf_repo_exists(url)


class TestSaveDataframeHfIntegration:
    """Tests for save_dataframe with HF URLs."""

    @patch("syntk.io._ensure_hf_repo_exists")
    @patch("pandas.DataFrame.to_csv")
    def test_calls_ensure_repo_for_hf_url(self, mock_to_csv, mock_ensure_repo):
        """Test that save_dataframe calls _ensure_hf_repo_exists for HF URLs."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        url = "hf://datasets/username/repo/file.tsv"

        save_dataframe(df, url)

        mock_ensure_repo.assert_called_once_with(url)
        mock_to_csv.assert_called_once()

    @patch("syntk.io._ensure_hf_repo_exists")
    @patch("pandas.DataFrame.to_parquet")
    def test_saves_to_hf_parquet(self, mock_to_parquet, mock_ensure_repo):
        """Test saving parquet file to HF."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        url = "hf://datasets/username/repo/file.parquet"

        save_dataframe(df, url)

        mock_ensure_repo.assert_called_once_with(url)
        mock_to_parquet.assert_called_once_with(url, index=False)

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_does_not_create_local_dirs_for_hf_url(self, mock_to_csv, mock_makedirs):
        """Test that local directories are not created for HF URLs."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        url = "hf://datasets/username/repo/file.csv"

        with patch("syntk.io._ensure_hf_repo_exists"):
            save_dataframe(df, url)

        mock_makedirs.assert_not_called()

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_creates_local_dirs_for_local_path(self, mock_to_csv, mock_makedirs):
        """Test that local directories are created for local paths."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        path = "/tmp/test/nested/file.csv"

        save_dataframe(df, path)

        mock_makedirs.assert_called_once_with("/tmp/test/nested", exist_ok=True)
