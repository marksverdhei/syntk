"""Base pipeline class for syntk data processing pipelines."""

import os
import sys
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from transformers import HfArgumentParser

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
    BaseDataArguments,
    BaseProcessingArguments,
)
from syntk.tracking import TrackingArguments, get_tracker, ExperimentTracker
from syntk.io import save_dataframe, load_dataframe

logger = logging.getLogger(__name__)
tqdm.pandas()


class BasePipeline(ABC):
    """Abstract base class for syntk pipelines.

    Provides common infrastructure:
    - Configuration parsing (CLI + YAML)
    - File I/O with resume capability
    - Checkpointing and progress saving
    - Experiment tracking integration
    - API client management
    - Metrics logging

    Subclasses must implement:
    - get_argument_classes(): Return tuple of dataclass types for argument parsing
    - setup_dataframe(): Initialize output columns
    - process_row(): Process a single row
    - get_config_params(): Return dict of config parameters to log
    - get_rows_to_process(): Return list of row indices that need processing
    """

    def __init__(self):
        """Initialize pipeline (called by subclass constructors)."""
        self.responses: Dict[str, Any] = {}  # Response cache
        self.client: Optional[OpenAI] = None
        self.tracker: Optional[ExperimentTracker] = None
        self.df: Optional[pd.DataFrame] = None

        # Arguments (set by parse_arguments)
        self.config_args: Optional[ConfigArguments] = None
        self.api_args: Optional[APIArguments] = None
        self.gen_args: Optional[GenerationArguments] = None
        self.data_args: Optional[BaseDataArguments] = None
        self.proc_args: Optional[BaseProcessingArguments] = None
        self.track_args: Optional[TrackingArguments] = None

    @abstractmethod
    def get_argument_classes(self) -> Tuple[Type, ...]:
        """Return tuple of argument dataclass types for parsing.

        Must include ConfigArguments, APIArguments, GenerationArguments,
        data args class, processing args class, and TrackingArguments.

        Example:
            return (ConfigArguments, APIArguments, GenerationArguments,
                    ColumnDataArguments, ColumnProcessingArguments, TrackingArguments)
        """
        pass

    @abstractmethod
    def setup_dataframe(self, df: pd.DataFrame, resuming: bool) -> pd.DataFrame:
        """Initialize output columns in dataframe.

        Args:
            df: Input dataframe
            resuming: Whether we're resuming from existing output

        Returns:
            Modified dataframe with output columns initialized
        """
        pass

    @abstractmethod
    def process_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """Process a single row.

        Args:
            row: DataFrame row as Series
            idx: Row index in dataframe

        Returns:
            Dict mapping column names to values to save
        """
        pass

    @abstractmethod
    def get_config_params(self) -> Dict[str, Any]:
        """Return configuration parameters to log to tracker.

        Returns:
            Dict of parameter names to values
        """
        pass

    @abstractmethod
    def get_rows_to_process(self, df: pd.DataFrame) -> List[int]:
        """Determine which rows need processing.

        Args:
            df: DataFrame to process

        Returns:
            List of row indices that need processing
        """
        pass

    def parse_arguments(self) -> None:
        """Parse command line arguments and YAML config with merging."""
        # Check if first positional argument is a YAML config file
        config_file = None
        args_to_parse = None
        if (
            len(sys.argv) > 1
            and sys.argv[1].endswith((".yaml", ".yml"))
            and not sys.argv[1].startswith("--")
        ):
            config_file = sys.argv[1]
            args_to_parse = sys.argv[2:]

        # Parse command line arguments
        arg_classes = self.get_argument_classes()
        parser = HfArgumentParser(arg_classes)
        parsed_args = parser.parse_args_into_dataclasses(args=args_to_parse)

        # Unpack into instance variables
        (
            self.config_args,
            self.api_args,
            self.gen_args,
            self.data_args,
            self.proc_args,
            self.track_args,
        ) = parsed_args

        # Set config file if provided as positional argument
        if config_file:
            self.config_args.config_file = config_file

        # If config file specified, load and merge with CLI args
        if self.config_args.config_file:
            self._merge_yaml_config(arg_classes[1:])  # Exclude ConfigArguments

    def _merge_yaml_config(self, arg_classes: Tuple[Type, ...]) -> None:
        """Load YAML config and merge with CLI arguments."""
        with open(self.config_args.config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse from flattened YAML dict
        yaml_parser = HfArgumentParser(arg_classes)
        yaml_args = yaml_parser.parse_dict(config_dict, allow_extra_keys=True)

        # Merge: use CLI value if it differs from default, otherwise use YAML value
        arg_objects = [
            self.api_args,
            self.gen_args,
            self.data_args,
            self.proc_args,
            self.track_args,
        ]

        for args_obj, yaml_obj in zip(arg_objects, yaml_args):
            for field_name in args_obj.__dataclass_fields__:
                cli_value = getattr(args_obj, field_name)
                yaml_value = getattr(yaml_obj, field_name)
                default_value = args_obj.__dataclass_fields__[field_name].default

                # If CLI value is still default, use YAML value
                if cli_value == default_value:
                    setattr(args_obj, field_name, yaml_value)

    def initialize_client(self) -> None:
        """Initialize OpenAI client from API arguments."""
        api_key = os.getenv(self.api_args.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {self.api_args.api_key_env}"
            )

        logger.info(
            f"Initializing OpenAI client with base_url: {self.api_args.base_url}, "
            f"model: {self.api_args.model}"
        )

        self.client = OpenAI(
            base_url=self.api_args.base_url,
            api_key=api_key,
        )

    def initialize_tracker(self) -> None:
        """Initialize experiment tracker and log config params."""
        self.tracker = get_tracker(self.track_args)

        # Log all configuration parameters
        config_params = self.get_config_params()
        self.tracker.log_params({k: v for k, v in config_params.items() if v is not None})

    def load_data(self) -> Tuple[pd.DataFrame, bool]:
        """Load data with resume capability.

        Returns:
            Tuple of (dataframe, resuming_flag)
        """
        resuming = False

        # Check if we can resume from existing output file
        if os.path.exists(self.data_args.output_file):
            try:
                logger.info(
                    f"Output file exists, checking for resume possibility: "
                    f"{self.data_args.output_file}"
                )
                df = load_dataframe(self.data_args.output_file)

                # Subclass determines if we can actually resume
                rows_to_process = self.get_rows_to_process(df)
                if len(rows_to_process) < len(df):
                    processed_count = len(df) - len(rows_to_process)
                    logger.info(
                        f"Found {processed_count}/{len(df)} rows already processed. "
                        f"Resuming..."
                    )
                    resuming = True
                else:
                    logger.info("Output file exists but no rows processed. Loading input file.")
                    df = load_dataframe(self.data_args.input_file)
            except Exception as e:
                logger.warning(
                    f"Could not load output file for resume: {e}. Loading input file instead."
                )
                df = load_dataframe(self.data_args.input_file)
        else:
            logger.info(f"Loading data from: {self.data_args.input_file}")
            df = load_dataframe(self.data_args.input_file)

        logger.info(f"Loaded {len(df)} samples")
        return df, resuming

    def apply_limit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sample limit if specified.

        Args:
            df: Input dataframe

        Returns:
            Limited dataframe
        """
        if self.data_args.limit is not None:
            if 0 < self.data_args.limit < 1:
                # Treat as fraction
                n_samples = int(len(df) * self.data_args.limit)
                logger.info(f"Limiting to {self.data_args.limit * 100}% of samples: {n_samples}")
                df = df.head(n_samples)
            elif self.data_args.limit >= 1:
                # Treat as absolute count
                logger.info(f"Limiting to {int(self.data_args.limit)} samples")
                df = df.head(int(self.data_args.limit))
            else:
                raise ValueError(
                    "Limit must be positive (integer for count, 0-1 for fraction)"
                )
        return df

    def run(self) -> None:
        """Main execution method - orchestrates entire pipeline."""
        # 1. Parse arguments
        self.parse_arguments()

        # 2. Initialize API client
        self.initialize_client()

        # 3. Initialize tracker
        self.initialize_tracker()

        # 4. Load data with resume capability
        self.df, resuming = self.load_data()

        # 5. Apply limit
        self.df = self.apply_limit(self.df)

        # 6. Setup dataframe (initialize columns)
        self.df = self.setup_dataframe(self.df, resuming)

        # 7. Identify rows to process
        rows_to_process = self.get_rows_to_process(self.df)

        if len(rows_to_process) == 0:
            logger.info("All rows already processed. Nothing to do.")
            return

        logger.info(f"Processing {len(rows_to_process)} rows...")

        # 8. Process rows
        self._process_rows(rows_to_process, resuming)

        # 9. Finish tracking
        self.tracker.finish()

    def _process_rows(self, rows_to_process: List[int], resuming: bool) -> None:
        """Process all rows with checkpointing and metrics.

        Args:
            rows_to_process: List of row indices to process
            resuming: Whether we're resuming from previous run
        """
        start_time = time.time()
        processed_count = 0
        initial_api_calls = len(self.responses)

        for idx in tqdm(rows_to_process, desc="Generating"):
            row = self.df.loc[idx]

            # Process row (implemented by subclass)
            result = self.process_row(row, idx)

            # Save results to dataframe
            for col_name, value in result.items():
                self.df.at[idx, col_name] = value

            processed_count += 1

            # Log metrics periodically
            if self.proc_args.log_interval > 0 and processed_count % self.proc_args.log_interval == 0:
                self._log_metrics(processed_count, initial_api_calls, start_time)

            # Save checkpoint
            if self.proc_args.save_interval > 0 and processed_count % self.proc_args.save_interval == 0:
                self._save_checkpoint(processed_count, len(rows_to_process))

        # Log final summary
        self._log_final_summary(processed_count, initial_api_calls, start_time)

        # Final save
        self._save_final_output()

    def _log_metrics(self, processed_count: int, initial_api_calls: int, start_time: float) -> None:
        """Log periodic metrics to tracker."""
        elapsed_time = time.time() - start_time
        self.tracker.log_metrics(
            {
                "rows_processed": processed_count,
                "total_api_calls": len(self.responses),
                "new_api_calls": len(self.responses) - initial_api_calls,
                "cache_hits": processed_count - (len(self.responses) - initial_api_calls),
                "elapsed_seconds": elapsed_time,
                "rows_per_second": processed_count / elapsed_time if elapsed_time > 0 else 0,
            },
            step=processed_count,
        )

    def _save_checkpoint(self, processed_count: int, total_count: int) -> None:
        """Save progress checkpoint."""
        logger.info(
            f"Checkpoint: Saving progress ({processed_count}/{total_count} processed)..."
        )
        save_dataframe(self.df, self.data_args.output_file)

    def _log_final_summary(self, processed_count: int, initial_api_calls: int, start_time: float) -> None:
        """Log final summary metrics."""
        total_time = time.time() - start_time
        total_api_calls = len(self.responses) - initial_api_calls
        cache_hit_rate = (
            (processed_count - total_api_calls) / processed_count
            if processed_count > 0
            else 0
        )

        self.tracker.log_summary(
            {
                "total_rows_processed": processed_count,
                "total_api_calls": total_api_calls,
                "total_cache_hits": processed_count - total_api_calls,
                "cache_hit_rate": cache_hit_rate,
                "total_time_seconds": total_time,
                "avg_time_per_row": total_time / processed_count if processed_count > 0 else 0,
            }
        )

    def _save_final_output(self) -> None:
        """Save final processed dataframe."""
        logger.info(f"Saving final processed data to {self.data_args.output_file}")
        save_dataframe(self.df, self.data_args.output_file)
        logger.info(f"Successfully saved {len(self.df)} processed samples")
        logger.info(f"Total unique API calls made: {len(self.responses)} (rest were cached)")
