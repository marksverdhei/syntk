"""Shared configuration dataclasses for syntk pipelines."""

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
class BaseDataArguments:
    """Base data arguments shared across all pipelines."""

    input_file: str = field(
        default="input.parquet",
        metadata={
            "help": "Input file path (supports .parquet, .csv, .json, .jsonl, .tsv)"
        },
    )
    output_file: str = field(
        default="output.parquet",
        metadata={
            "help": "Output file path (supports .parquet, .csv, .json, .jsonl, .tsv)"
        },
    )
    limit: Optional[float] = field(
        default=None,
        metadata={
            "help": "Limit samples: integer for count, 0-1 for fraction, None for all"
        },
    )


@dataclass
class BaseProcessingArguments:
    """Base processing arguments shared across all pipelines."""

    log_interval: int = field(
        default=10,
        metadata={"help": "Log metrics every N rows (0 to only log at end)"},
    )
    save_interval: int = field(
        default=100,
        metadata={"help": "Save progress every N rows (0 to only save at end)"},
    )
