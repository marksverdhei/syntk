"""Column pipeline - fill column values using LLM."""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
    BaseDataArguments,
    BaseProcessingArguments,
)
from syntk.tracking import TrackingArguments
from syntk.pipelines.base import BasePipeline
from syntk.api_client import get_chat_response, save_raw_api_call
from syntk.io import find_available_column_name

logger = logging.getLogger(__name__)


@dataclass
class ColumnDataArguments(BaseDataArguments):
    """Data arguments specific to column pipeline."""

    text_column: str = field(
        default="text", metadata={"help": "Name of the text column in the dataset"}
    )
    output_column: str = field(
        default="generated",
        metadata={"help": "Name of the generated output column"},
    )
    reasoning_content_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional column name to save reasoning traces (for reasoning models)"
        },
    )
    save_stop_reason: bool = field(
        default=False,
        metadata={"help": "Save stop/finish reason to a column named 'stop_reason'"},
    )
    raw_api_json_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to save raw API requests/responses as JSONL (one JSON per line)"
        },
    )


@dataclass
class ColumnProcessingArguments(BaseProcessingArguments):
    """Processing arguments specific to column pipeline."""

    prompt_template: str = field(
        default="Process the following text:\n\n{text}",
        metadata={
            "help": "Prompt template with placeholders for column names (e.g., {text}, {label})"
        },
    )


class ColumnPipeline(BasePipeline):
    """Pipeline for filling column values using LLM.

    Features:
    - Flexible prompt templates with column interpolation
    - Response caching to avoid redundant API calls
    - Optional reasoning content capture
    - Optional stop reason tracking
    - Raw API request/response logging
    """

    def __init__(self):
        super().__init__()
        self.actual_stop_reason_column: Optional[str] = None

    def get_argument_classes(self) -> Tuple[Type, ...]:
        """Return argument classes for column pipeline."""
        return (
            ConfigArguments,
            APIArguments,
            GenerationArguments,
            ColumnDataArguments,
            ColumnProcessingArguments,
            TrackingArguments,
        )

    def get_config_params(self) -> Dict[str, Any]:
        """Return config parameters to log."""
        return {
            "model": self.api_args.model,
            "base_url": self.api_args.base_url,
            "temperature": self.gen_args.temperature,
            "max_tokens": self.gen_args.max_tokens,
            "top_p": self.gen_args.top_p,
            "frequency_penalty": self.gen_args.frequency_penalty,
            "presence_penalty": self.gen_args.presence_penalty,
            "input_file": self.data_args.input_file,
            "output_file": self.data_args.output_file,
            "text_column": self.data_args.text_column,
            "output_column": self.data_args.output_column,
            "limit": self.data_args.limit,
            "log_interval": self.proc_args.log_interval,
            "save_interval": self.proc_args.save_interval,
        }

    def setup_dataframe(self, df: pd.DataFrame, resuming: bool) -> pd.DataFrame:
        """Initialize output columns."""
        # Initialize output column if not resuming
        if not resuming or self.data_args.output_column not in df.columns:
            df[self.data_args.output_column] = pd.NA

        # Initialize optional stop reason column
        if self.data_args.save_stop_reason:
            self.actual_stop_reason_column = find_available_column_name(df, "stop_reason")
            if not resuming or self.actual_stop_reason_column not in df.columns:
                df[self.actual_stop_reason_column] = pd.NA
            logger.info(f"Stop reasons will be saved to column: {self.actual_stop_reason_column}")

        # Initialize optional reasoning content column
        if self.data_args.reasoning_content_column:
            if not resuming or self.data_args.reasoning_content_column not in df.columns:
                df[self.data_args.reasoning_content_column] = pd.NA
            logger.info(
                f"Reasoning content will be saved to column: {self.data_args.reasoning_content_column}"
            )

        # Initialize raw API JSON file if specified (clear file if not resuming)
        if self.data_args.raw_api_json_path and not resuming:
            raw_dir = os.path.dirname(self.data_args.raw_api_json_path)
            if raw_dir:
                os.makedirs(raw_dir, exist_ok=True)
            # Create empty file (or truncate if exists)
            open(self.data_args.raw_api_json_path, "w").close()
            logger.info(f"Raw API data will be saved to: {self.data_args.raw_api_json_path}")

        return df

    def get_rows_to_process(self, df: pd.DataFrame) -> List[int]:
        """Identify rows with missing values in output column."""
        if self.data_args.output_column not in df.columns:
            return df.index.tolist()
        return df[df[self.data_args.output_column].isna()].index.tolist()

    def process_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """Process a single row by generating text based on prompt template.

        Returns:
            Dict mapping column names to values
        """
        # Create a dictionary of all column values for formatting
        format_dict = row.to_dict()

        # Format the prompt using all available columns
        prompt = self.proc_args.prompt_template.format(**format_dict)

        # Use cache if available
        if prompt in self.responses:
            logger.debug("Using cached response for prompt")
            api_result = self.responses[prompt]
        else:
            logger.debug("Making new API call (cache miss)")
            return_raw = self.data_args.raw_api_json_path is not None
            api_result = get_chat_response(
                client=self.client,
                prompt=prompt,
                model=self.api_args.model,
                temperature=self.gen_args.temperature,
                max_tokens=self.gen_args.max_tokens,
                top_p=self.gen_args.top_p,
                frequency_penalty=self.gen_args.frequency_penalty,
                presence_penalty=self.gen_args.presence_penalty,
                return_raw=return_raw,
            )
            self.responses[prompt] = api_result

        # Build result dict with columns to save
        result = {self.data_args.output_column: api_result["content"]}

        # Add reasoning content if column specified
        if self.data_args.reasoning_content_column:
            result[self.data_args.reasoning_content_column] = api_result[
                "reasoning_content"
            ]

        # Add stop reason if enabled
        if self.actual_stop_reason_column:
            result[self.actual_stop_reason_column] = api_result["stop_reason"]

        # Save raw API request/response if enabled
        if self.data_args.raw_api_json_path:
            save_raw_api_call(self.data_args.raw_api_json_path, idx, api_result)

        return result


def main() -> None:
    """Main entry point for column pipeline."""
    pipeline = ColumnPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
