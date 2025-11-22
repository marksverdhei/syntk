"""Tests for pipelines.column data I/O functionality."""
import os
import tempfile
import shutil
import pandas as pd
import pytest
from syntk.pipelines.column import save_dataframe, load_dataframe


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe with various data types."""
    return pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'nullable_col': [1, None, 3]
    })


class TestSaveDataframe:
    """Tests for save_dataframe function."""

    def test_creates_nested_directory(self, temp_dir, sample_dataframe):
        """Test that save_dataframe creates nested directories."""
        output_path = os.path.join(temp_dir, "nested", "dirs", "output.csv")

        save_dataframe(sample_dataframe, output_path)

        assert os.path.exists(output_path)
        assert os.path.isdir(os.path.join(temp_dir, "nested", "dirs"))

    def test_saves_to_current_directory(self, temp_dir, sample_dataframe):
        """Test saving file without directory component."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            output_path = "output.csv"

            save_dataframe(sample_dataframe, output_path)

            assert os.path.exists(output_path)
        finally:
            os.chdir(original_cwd)

    def test_overwrites_existing_file(self, temp_dir, sample_dataframe):
        """Test that save_dataframe overwrites existing files."""
        output_path = os.path.join(temp_dir, "output.csv")

        # Save once
        save_dataframe(sample_dataframe, output_path)
        first_content = pd.read_csv(output_path)

        # Save again with different data
        modified_df = sample_dataframe.copy()
        modified_df['int_col'] = [10, 20, 30]
        save_dataframe(modified_df, output_path)
        second_content = pd.read_csv(output_path)

        assert not first_content['int_col'].equals(second_content['int_col'])

    def test_rejects_unsupported_format(self, temp_dir, sample_dataframe):
        """Test that unsupported formats raise ValueError."""
        output_path = os.path.join(temp_dir, "output.xlsx")

        with pytest.raises(ValueError, match="Unsupported output format"):
            save_dataframe(sample_dataframe, output_path)


class TestLoadDataframe:
    """Tests for load_dataframe function."""

    def test_rejects_unsupported_format(self, temp_dir):
        """Test that unsupported formats raise ValueError."""
        input_path = os.path.join(temp_dir, "input.xlsx")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(input_path)


class TestRoundtrips:
    """Tests for save/load roundtrips across different formats."""

    def test_csv_roundtrip(self, temp_dir, sample_dataframe):
        """Test CSV save and load preserves data."""
        output_path = os.path.join(temp_dir, "test.csv")

        save_dataframe(sample_dataframe, output_path)
        loaded_df = load_dataframe(output_path)

        # CSV doesn't preserve exact dtypes, so compare values
        pd.testing.assert_frame_equal(
            sample_dataframe.fillna(''),
            loaded_df.fillna(''),
            check_dtype=False
        )

    def test_json_roundtrip(self, temp_dir, sample_dataframe):
        """Test JSON save and load preserves data."""
        output_path = os.path.join(temp_dir, "test.json")

        save_dataframe(sample_dataframe, output_path)
        loaded_df = load_dataframe(output_path)

        pd.testing.assert_frame_equal(
            sample_dataframe,
            loaded_df,
            check_dtype=False
        )

    def test_jsonl_roundtrip(self, temp_dir, sample_dataframe):
        """Test JSONL save and load preserves data."""
        output_path = os.path.join(temp_dir, "test.jsonl")

        save_dataframe(sample_dataframe, output_path)
        loaded_df = load_dataframe(output_path)

        pd.testing.assert_frame_equal(
            sample_dataframe,
            loaded_df,
            check_dtype=False
        )

    def test_tsv_roundtrip(self, temp_dir, sample_dataframe):
        """Test TSV save and load preserves data."""
        output_path = os.path.join(temp_dir, "test.tsv")

        save_dataframe(sample_dataframe, output_path)
        loaded_df = load_dataframe(output_path)

        pd.testing.assert_frame_equal(
            sample_dataframe.fillna(''),
            loaded_df.fillna(''),
            check_dtype=False
        )

    def test_parquet_roundtrip(self, temp_dir, sample_dataframe):
        """Test Parquet save and load preserves data."""
        output_path = os.path.join(temp_dir, "test.parquet")

        save_dataframe(sample_dataframe, output_path)
        loaded_df = load_dataframe(output_path)

        pd.testing.assert_frame_equal(
            sample_dataframe,
            loaded_df,
            check_dtype=False
        )

    def test_format_detection_case_insensitive(self, temp_dir, sample_dataframe):
        """Test that format detection works with different case extensions."""
        for ext in ['.CSV', '.Json', '.JSONL', '.Tsv', '.Parquet']:
            output_path = os.path.join(temp_dir, f"test{ext}")
            save_dataframe(sample_dataframe, output_path)
            loaded_df = load_dataframe(output_path)
            assert len(loaded_df) == len(sample_dataframe)

    def test_preserves_column_names_with_special_chars(self, temp_dir):
        """Test that column names with special characters are preserved."""
        df = pd.DataFrame({
            'column with spaces': [1, 2, 3],
            'column-with-dashes': [4, 5, 6],
            'column.with.dots': [7, 8, 9]
        })
        output_path = os.path.join(temp_dir, "test.csv")

        save_dataframe(df, output_path)
        loaded_df = load_dataframe(output_path)

        assert list(loaded_df.columns) == list(df.columns)
