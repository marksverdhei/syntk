"""Argument dataclasses for syntk.

Provides configuration dataclasses for various pipeline components.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfigArguments:
    """Arguments for configuration file."""

    config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to YAML config file. If provided, loads defaults from file."
        },
    )


@dataclass
class APIArguments:
    """Arguments for API configuration."""

    base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL for OpenAI-compatible API"},
    )
    api_key_env: str = field(
        default="OPENAI_API_KEY",
        metadata={"help": "Environment variable name for API key"},
    )
    model: str = field(default="gpt-3.5-turbo", metadata={"help": "Model name to use"})


@dataclass
class GenerationArguments:
    """Arguments for text generation."""

    temperature: Optional[float] = field(
        default=None, metadata={"help": "Sampling temperature"}
    )
    max_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum tokens to generate"}
    )
    top_p: Optional[float] = field(
        default=None, metadata={"help": "Nucleus sampling probability"}
    )
    frequency_penalty: Optional[float] = field(
        default=None, metadata={"help": "Frequency penalty"}
    )
    presence_penalty: Optional[float] = field(
        default=None, metadata={"help": "Presence penalty"}
    )


@dataclass
class DataArguments:
    """Arguments for data input/output."""

    input_file: str = field(
        default="hf://datasets/ltg/norec_sentence/ternary/train-00000-of-00001.parquet",
        metadata={"help": "Input parquet file path"},
    )
    output_file: str = field(
        default="annotated_difficulty.parquet",
        metadata={"help": "Output parquet file path"},
    )
    text_column: str = field(
        default="review", metadata={"help": "Name of the text column in the dataset"}
    )
    output_column: str = field(
        default="difficulty_annotation",
        metadata={"help": "Name of the generated output column"},
    )
    limit: Optional[float] = field(
        default=None,
        metadata={
            "help": "Limit samples: integer for count, 0-1 for fraction, None for all"
        },
    )


@dataclass
class ProcessingArguments:
    """Arguments for processing configuration."""

    prompt_template: str = field(
        default="""Analyze the difficulty of classifying the sentiment of the following Norwegian text:

Text: {text}
True sentiment: {sentiment}

Rate the difficulty on a scale of 1-5 and provide a brief explanation.""",
        metadata={"help": "Prompt template with {text} and {sentiment} placeholders"},
    )
    save_interval: int = field(
        default=100,
        metadata={"help": "Save progress every N rows (0 to only save at end)"},
    )
