"""End-to-end tests for column pipeline with mocked OpenAI API."""

import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch
import pandas as pd
import pytest
from openai import AuthenticationError
from syntk.pipelines.column import (
    main,
    ColumnPipeline,
    ColumnDataArguments as DataArguments,
    ColumnProcessingArguments as ProcessingArguments,
)
from syntk.api_client import get_chat_response
from syntk.io import save_dataframe, load_dataframe
from syntk.config import APIArguments, GenerationArguments


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def sample_input_data(temp_dir):
    """Create sample input data file."""
    df = pd.DataFrame(
        {
            "text": [
                "What is the capital of France?",
                "Explain quantum computing",
                "Write a haiku about programming",
            ],
            "id": [1, 2, 3],
        }
    )
    input_file = os.path.join(temp_dir, "input.parquet")
    df.to_parquet(input_file, index=False)
    return input_file, df


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with compliant response objects."""
    mock_client = Mock()

    def create_mock_response(content, reasoning_content=None, finish_reason="stop"):
        """Create OpenAI-compliant response object."""
        # Create mock message
        mock_message = Mock()
        mock_message.content = content
        mock_message.reasoning_content = reasoning_content

        # Create mock choice
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = finish_reason

        # Create mock usage
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        # Create mock response
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model_dump = Mock(
            return_value={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        )

        return mock_response

    # Set up responses for different prompts
    responses = [
        create_mock_response("The capital of France is Paris."),
        create_mock_response(
            "Quantum computing uses quantum bits (qubits) that can exist in superposition."
        ),
        create_mock_response(
            "Code flows like streams,\nBugs hide in silent shadows,\nDebug brings the dawn."
        ),
    ]

    mock_client.chat.completions.create = Mock(side_effect=responses)

    return mock_client


@pytest.fixture
def mock_openai_client_with_reasoning():
    """Create a mock OpenAI client with reasoning content."""
    mock_client = Mock()

    def create_mock_response_with_reasoning(
        content, reasoning_content, finish_reason="stop"
    ):
        """Create OpenAI-compliant response object with reasoning."""
        mock_message = Mock()
        mock_message.content = content
        mock_message.reasoning_content = reasoning_content

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = finish_reason

        mock_usage = Mock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 250

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model_dump = Mock(
            return_value={
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "o1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning_content,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 100,
                    "total_tokens": 250,
                },
            }
        )

        return mock_response

    response = create_mock_response_with_reasoning(
        "42",
        "Let me think step by step: The question asks for the answer to life, the universe, and everything.",
    )

    mock_client.chat.completions.create = Mock(return_value=response)

    return mock_client


class TestGetChatResponse:
    """Tests for get_chat_response function with mocked API."""

    def test_basic_chat_response(self, mock_openai_client):
        """Test basic chat response with mocked OpenAI client."""
        result = get_chat_response(
            client=mock_openai_client,
            prompt="What is 2+2?",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
        )

        assert result["content"] is not None
        assert "reasoning_content" in result
        assert result["stop_reason"] == "stop"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_chat_response_with_generation_params(self, mock_openai_client):
        """Test that generation parameters are passed correctly."""
        get_chat_response(
            client=mock_openai_client,
            prompt="Test prompt",
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=200,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 200
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["frequency_penalty"] == 0.5
        assert call_kwargs["presence_penalty"] == 0.3

    def test_chat_response_with_raw_data(self, mock_openai_client):
        """Test that raw request/response data is returned when requested."""
        result = get_chat_response(
            client=mock_openai_client,
            prompt="Test prompt",
            model="gpt-3.5-turbo",
            return_raw=True,
        )

        assert "raw" in result
        assert "request" in result["raw"]
        assert "response" in result["raw"]
        assert result["raw"]["request"]["model"] == "gpt-3.5-turbo"

    def test_chat_response_with_reasoning_content(
        self, mock_openai_client_with_reasoning
    ):
        """Test chat response with reasoning content from reasoning models."""
        result = get_chat_response(
            client=mock_openai_client_with_reasoning,
            prompt="What is the answer?",
            model="o1-mini",
        )

        assert result["content"] == "42"
        assert result["reasoning_content"] is not None
        assert "step by step" in result["reasoning_content"]

    def test_chat_response_authentication_error(self, caplog):
        """Test that authentication errors are properly caught and reported."""
        from openai import AuthenticationError as OpenAIAuthError

        mock_client = Mock()
        # Simulate an authentication error
        mock_client.chat.completions.create = Mock(
            side_effect=OpenAIAuthError(
                "Invalid API key", response=Mock(status_code=401), body=None
            )
        )

        with pytest.raises(AuthenticationError):
            get_chat_response(
                client=mock_client,
                prompt="Test prompt",
                model="gpt-3.5-turbo",
            )

        # Verify helpful error message was logged
        assert "Authentication failed" in caplog.text
        assert "API key is missing or invalid" in caplog.text


class TestProcessRow:
    """Tests for process_row method."""

    def test_process_row_basic(self, mock_openai_client):
        """Test processing a single row with mocked API."""
        row = pd.Series({"text": "Hello world", "id": 1})

        # Create a pipeline instance with minimal setup
        pipeline = ColumnPipeline()
        pipeline.client = mock_openai_client
        pipeline.api_args = APIArguments(model="gpt-3.5-turbo")
        pipeline.gen_args = GenerationArguments()
        pipeline.data_args = DataArguments(
            text_column="text", output_column="generated"
        )
        pipeline.proc_args = ProcessingArguments(prompt_template="Process: {text}")

        result = pipeline.process_row(row, 0)

        assert result["generated"] is not None
        assert len(pipeline.responses) == 1

    def test_process_row_caching(self, mock_openai_client):
        """Test that identical prompts use cached responses."""
        row = pd.Series({"text": "Same prompt", "id": 1})

        # Create a pipeline instance with minimal setup
        pipeline = ColumnPipeline()
        pipeline.client = mock_openai_client
        pipeline.api_args = APIArguments(model="gpt-3.5-turbo")
        pipeline.gen_args = GenerationArguments()
        pipeline.data_args = DataArguments(
            text_column="text", output_column="generated"
        )
        pipeline.proc_args = ProcessingArguments(prompt_template="{text}")

        # First call
        result1 = pipeline.process_row(row, 0)

        # Second call with same prompt
        result2 = pipeline.process_row(row, 0)

        assert result1 == result2
        # Should only call API once due to caching
        assert mock_openai_client.chat.completions.create.call_count == 1
        assert len(pipeline.responses) == 1


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline with file I/O."""

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("syntk.pipelines.base.get_tracker")
    def test_complete_pipeline_with_file_writing(
        self,
        mock_tracker,
        mock_openai_class,
        temp_dir,
        sample_input_data,
        mock_openai_client,
    ):
        """Test complete pipeline from input file to output file."""
        input_file, input_df = sample_input_data
        output_file = os.path.join(temp_dir, "output.parquet")

        # Configure mock
        mock_openai_class.return_value = mock_openai_client
        mock_tracker.return_value = Mock()

        # Patch sys.argv to simulate command line arguments
        test_args = [
            "syntk",
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--text_column",
            "text",
            "--output_column",
            "generated",
            "--model",
            "gpt-3.5-turbo",
            "--base_url",
            "http://localhost:8000/v1",
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify output file was created
        assert os.path.exists(output_file)

        # Load and verify output
        output_df = load_dataframe(output_file)
        assert len(output_df) == len(input_df)
        assert "generated" in output_df.columns
        assert output_df["generated"].notna().all()

        # Verify API was called correct number of times
        assert mock_openai_client.chat.completions.create.call_count == 3

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("syntk.pipelines.base.get_tracker")
    def test_pipeline_with_reasoning_content(
        self,
        mock_tracker,
        mock_openai_class,
        temp_dir,
        mock_openai_client_with_reasoning,
    ):
        """Test pipeline saves reasoning content when specified."""
        # Create input data
        df = pd.DataFrame({"text": ["What is the answer?"]})
        input_file = os.path.join(temp_dir, "input.csv")
        df.to_csv(input_file, index=False)

        output_file = os.path.join(temp_dir, "output.csv")

        mock_openai_class.return_value = mock_openai_client_with_reasoning
        mock_tracker.return_value = Mock()

        test_args = [
            "syntk",
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--text_column",
            "text",
            "--output_column",
            "answer",
            "--reasoning_content_column",
            "reasoning",
            "--model",
            "o1-mini",
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify reasoning content was saved
        output_df = load_dataframe(output_file)
        assert "reasoning" in output_df.columns
        assert output_df["reasoning"].notna().all()
        assert "step by step" in output_df["reasoning"].iloc[0]

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("syntk.pipelines.base.get_tracker")
    def test_pipeline_with_stop_reason(
        self,
        mock_tracker,
        mock_openai_class,
        temp_dir,
        sample_input_data,
        mock_openai_client,
    ):
        """Test pipeline saves stop reason when specified."""
        input_file, _ = sample_input_data
        output_file = os.path.join(temp_dir, "output.json")

        mock_openai_class.return_value = mock_openai_client
        mock_tracker.return_value = Mock()

        test_args = [
            "syntk",
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--save_stop_reason",
            "--text_column",
            "text",
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify stop reason was saved
        output_df = load_dataframe(output_file)
        assert "stop_reason" in output_df.columns
        assert (output_df["stop_reason"] == "stop").all()

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("syntk.pipelines.base.get_tracker")
    def test_pipeline_with_raw_api_json(
        self,
        mock_tracker,
        mock_openai_class,
        temp_dir,
        sample_input_data,
        mock_openai_client,
    ):
        """Test pipeline saves raw API requests/responses to JSONL."""
        input_file, input_df = sample_input_data
        output_file = os.path.join(temp_dir, "output.parquet")
        raw_json_file = os.path.join(temp_dir, "raw_api.jsonl")

        mock_openai_class.return_value = mock_openai_client
        mock_tracker.return_value = Mock()

        test_args = [
            "syntk",
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--raw_api_json_path",
            raw_json_file,
            "--text_column",
            "text",
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify raw JSON file was created
        assert os.path.exists(raw_json_file)

        # Load and verify raw JSON content
        with open(raw_json_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == len(input_df)

        # Parse each line and verify structure
        for line in lines:
            record = json.loads(line)
            assert "timestamp" in record
            assert "row_index" in record
            assert "request" in record
            assert "response" in record
            assert record["request"]["model"] == "gpt-3.5-turbo"
            assert "messages" in record["request"]

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("syntk.pipelines.base.get_tracker")
    def test_pipeline_multiple_file_formats(
        self, mock_tracker, mock_openai_class, temp_dir
    ):
        """Test pipeline works with different file formats."""
        formats = ["csv", "json", "jsonl", "tsv", "parquet"]

        for fmt in formats:
            # Create input file in this format
            df = pd.DataFrame({"text": [f"Test {fmt}"], "id": [1]})
            input_file = os.path.join(temp_dir, f"input.{fmt}")

            if fmt == "csv":
                df.to_csv(input_file, index=False)
            elif fmt == "json":
                df.to_json(input_file, orient="records", lines=False)
            elif fmt == "jsonl":
                df.to_json(input_file, orient="records", lines=True)
            elif fmt == "tsv":
                df.to_csv(input_file, sep="\t", index=False)
            elif fmt == "parquet":
                df.to_parquet(input_file, index=False)

            output_file = os.path.join(temp_dir, f"output.{fmt}")

            # Create a fresh mock client for each format
            mock_client = Mock()

            # Create mock response
            mock_message = Mock()
            mock_message.content = f"Processed: Test {fmt}"
            mock_message.reasoning_content = None

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"

            mock_usage = Mock()
            mock_usage.prompt_tokens = 100
            mock_usage.completion_tokens = 50
            mock_usage.total_tokens = 150

            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_response.model_dump = Mock(
                return_value={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "gpt-3.5-turbo",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Processed: Test {fmt}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                }
            )

            mock_client.chat.completions.create = Mock(return_value=mock_response)
            mock_openai_class.return_value = mock_client
            mock_tracker.return_value = Mock()

            test_args = [
                "syntk",
                "--input_file",
                input_file,
                "--output_file",
                output_file,
                "--text_column",
                "text",
            ]

            with patch("sys.argv", test_args):
                main()

            # Verify output file exists and has correct data
            assert os.path.exists(output_file), f"Output file not created for {fmt}"
            output_df = load_dataframe(output_file)
            assert len(output_df) == 1, f"Wrong row count for {fmt}"
            assert "generated" in output_df.columns, (
                f"Missing generated column for {fmt}"
            )

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("syntk.pipelines.base.get_tracker")
    def test_pipeline_resume_functionality(
        self,
        mock_tracker,
        mock_openai_class,
        temp_dir,
        sample_input_data,
        mock_openai_client,
    ):
        """Test that pipeline can resume from partially completed output file."""
        input_file, input_df = sample_input_data
        output_file = os.path.join(temp_dir, "output.parquet")

        # Create partially completed output file (only first row processed)
        partial_df = input_df.copy()
        partial_df["generated"] = pd.NA
        partial_df.at[0, "generated"] = "Already processed"
        save_dataframe(partial_df, output_file)

        mock_openai_class.return_value = mock_openai_client
        mock_tracker.return_value = Mock()

        test_args = [
            "syntk",
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--text_column",
            "text",
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify only remaining rows were processed
        # Should only call API 2 times (for rows 2 and 3)
        assert mock_openai_client.chat.completions.create.call_count == 2

        # Verify output
        output_df = load_dataframe(output_file)
        assert output_df.at[0, "generated"] == "Already processed"
        assert output_df["generated"].notna().all()

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("syntk.pipelines.base.get_tracker")
    def test_pipeline_with_limit(
        self,
        mock_tracker,
        mock_openai_class,
        temp_dir,
        sample_input_data,
        mock_openai_client,
    ):
        """Test pipeline respects limit parameter."""
        input_file, _ = sample_input_data
        output_file = os.path.join(temp_dir, "output.parquet")

        mock_openai_class.return_value = mock_openai_client
        mock_tracker.return_value = Mock()

        test_args = [
            "syntk",
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--text_column",
            "text",
            "--limit",
            "2",
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify only 2 rows were processed
        output_df = load_dataframe(output_file)
        assert len(output_df) == 2
        assert mock_openai_client.chat.completions.create.call_count == 2

    @patch("syntk.pipelines.base.OpenAI")
    @patch.dict(os.environ, {}, clear=True)  # Clear all env vars including API keys
    @patch("syntk.pipelines.base.get_tracker")
    def test_pipeline_without_api_key_for_local_api(
        self,
        mock_tracker,
        mock_openai_class,
        temp_dir,
        sample_input_data,
        mock_openai_client,
    ):
        """Test pipeline works without API key for local APIs (uses placeholder)."""
        input_file, input_df = sample_input_data
        output_file = os.path.join(temp_dir, "output.parquet")

        mock_openai_class.return_value = mock_openai_client
        mock_tracker.return_value = Mock()

        test_args = [
            "syntk",
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--text_column",
            "text",
            "--base_url",
            "http://localhost:8000/v1",  # Local API
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify pipeline completed successfully
        assert os.path.exists(output_file)
        output_df = load_dataframe(output_file)
        assert len(output_df) == len(input_df)
        assert "generated" in output_df.columns
        assert output_df["generated"].notna().all()

        # Verify OpenAI client was created with placeholder key
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "placeholder-api-key"
        assert call_kwargs["base_url"] == "http://localhost:8000/v1"
