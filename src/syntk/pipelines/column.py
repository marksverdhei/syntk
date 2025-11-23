import logging
import os
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from syntk.api import get_chat_response
from syntk.argument_parser import HfArgumentParser
from syntk.arguments import (
    APIArguments,
    ConfigArguments,
    DataArguments,
    GenerationArguments,
    ProcessingArguments,
)
from syntk.data_io import load_dataframe, save_dataframe
from syntk.tracking import TrackingArguments, get_tracker

tqdm.pandas()

logger = logging.getLogger(__name__)


def annotate_difficulty(
    row,
    client: OpenAI,
    api_args: APIArguments,
    gen_args: GenerationArguments,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
    responses: dict,
) -> str:
    """Annotate a single row with difficulty rating."""
    # Create a dictionary of all column values for formatting
    # This allows any column to be referenced in the prompt template
    format_dict = row.to_dict()

    # Format the prompt using all available columns
    prompt = proc_args.prompt_template.format(**format_dict)

    # Use cache if available
    if prompt in responses:
        logger.debug("Using cached response for prompt")
        return responses[prompt]

    logger.debug("Making new API call (cache miss)")
    response = get_chat_response(client, prompt, api_args, gen_args)
    responses[prompt] = response
    return response


def main() -> None:
    """Main function to annotate difficulty ratings."""
    import sys

    import yaml

    # Initialize response cache (local to this run)
    responses = {}

    # Check if first positional argument is a YAML config file
    config_file = None
    args_to_parse = None
    if (
        len(sys.argv) > 1
        and sys.argv[1].endswith((".yaml", ".yml"))
        and not sys.argv[1].startswith("--")
    ):
        config_file = sys.argv[1]
        args_to_parse = sys.argv[2:]  # Parse remaining args after the config file

    # Parse command line arguments
    parser = HfArgumentParser(
        (
            ConfigArguments,
            APIArguments,
            GenerationArguments,
            DataArguments,
            ProcessingArguments,
            TrackingArguments,
        )
    )
    config_args, api_args, gen_args, data_args, proc_args, track_args = (
        parser.parse_args_into_dataclasses(args=args_to_parse)
    )

    # Set config file if it was provided as positional argument
    if config_file:
        config_args.config_file = config_file

    # If config file is specified, load it and merge with CLI args
    if config_args.config_file:
        with open(config_args.config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse from flattened YAML dict
        yaml_parser = HfArgumentParser(
            (APIArguments, GenerationArguments, DataArguments, ProcessingArguments, TrackingArguments)
        )
        api_args_yaml, gen_args_yaml, data_args_yaml, proc_args_yaml, track_args_yaml = (
            yaml_parser.parse_dict(config_dict, allow_extra_keys=True)
        )

        # For each argument, use CLI value if it differs from default, otherwise use YAML value
        for args_obj, yaml_obj in [
            (api_args, api_args_yaml),
            (gen_args, gen_args_yaml),
            (data_args, data_args_yaml),
            (proc_args, proc_args_yaml),
            (track_args, track_args_yaml),
        ]:
            for field_name in args_obj.__dataclass_fields__:
                cli_value = getattr(args_obj, field_name)
                yaml_value = getattr(yaml_obj, field_name)
                default_value = args_obj.__dataclass_fields__[field_name].default

                # If CLI value is still default, use YAML value
                if cli_value == default_value:
                    setattr(args_obj, field_name, yaml_value)

    # Get API key from environment variable
    api_key = os.getenv(api_args.api_key_env)
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {api_args.api_key_env}"
        )

    logger.info(
        f"Initializing OpenAI client with base_url: {api_args.base_url}, model: {api_args.model}"
    )

    # Initialize OpenAI client
    client = OpenAI(
        base_url=api_args.base_url,
        api_key=api_key,
    )

    # Initialize experiment tracker
    tracker = get_tracker(track_args)

    # Log all configuration parameters
    config_params = {
        "model": api_args.model,
        "base_url": api_args.base_url,
        "temperature": gen_args.temperature,
        "max_tokens": gen_args.max_tokens,
        "top_p": gen_args.top_p,
        "frequency_penalty": gen_args.frequency_penalty,
        "presence_penalty": gen_args.presence_penalty,
        "input_file": data_args.input_file,
        "output_file": data_args.output_file,
        "text_column": data_args.text_column,
        "output_column": data_args.output_column,
        "limit": data_args.limit,
        "save_interval": proc_args.save_interval,
    }
    tracker.log_params({k: v for k, v in config_params.items() if v is not None})

    # Check if we can resume from existing output file
    resuming = False
    if os.path.exists(data_args.output_file):
        try:
            logger.info(
                f"Output file exists, checking for resume possibility: {data_args.output_file}"
            )
            df = load_dataframe(data_args.output_file)

            if data_args.output_column in df.columns:
                # Count how many rows are already processed
                processed_count = df[data_args.output_column].notna().sum()
                total_count = len(df)
                logger.info(
                    f"Found {processed_count}/{total_count} rows already processed. Resuming..."
                )
                resuming = True
            else:
                logger.info(
                    f"Output column '{data_args.output_column}' not found in output file. Loading input file instead."
                )
                df = load_dataframe(data_args.input_file)
        except Exception as e:
            logger.warning(
                f"Could not load output file for resume: {e}. Loading input file instead."
            )
            df = load_dataframe(data_args.input_file)
    else:
        logger.info(f"Loading data from: {data_args.input_file}")
        df = load_dataframe(data_args.input_file)

    logger.info(f"Loaded {len(df)} samples")

    # Apply limit if specified
    if data_args.limit is not None:
        if data_args.limit > 0 and data_args.limit < 1:
            # Treat as fraction
            n_samples = int(len(df) * data_args.limit)
            logger.info(f"Limiting to {data_args.limit * 100}% of samples: {n_samples}")
            df = df.head(n_samples)
        elif data_args.limit >= 1:
            # Treat as absolute count
            logger.info(f"Limiting to {int(data_args.limit)} samples")
            df = df.head(int(data_args.limit))
        else:
            raise ValueError(
                "Limit must be positive (integer for count, 0-1 for fraction)"
            )

    # Initialize output column if not resuming
    if not resuming or data_args.output_column not in df.columns:
        df[data_args.output_column] = pd.NA

    # Identify rows to process (those with missing values in output column)
    rows_to_process = df[df[data_args.output_column].isna()].index.tolist()

    if len(rows_to_process) == 0:
        logger.info("All rows already processed. Nothing to do.")
        return

    logger.info(f"Processing {len(rows_to_process)} rows...")

    # Track start time
    start_time = time.time()

    # Process rows one at a time with periodic checkpointing
    processed_count = 0
    initial_api_calls = len(responses)

    for idx in tqdm(rows_to_process, desc="Generating"):
        row = df.loc[idx]
        result = annotate_difficulty(
            row, client, api_args, gen_args, data_args, proc_args, responses
        )
        df.at[idx, data_args.output_column] = result
        processed_count += 1

        # Log metrics periodically
        if processed_count % 10 == 0:
            elapsed_time = time.time() - start_time
            tracker.log_metrics(
                {
                    "rows_processed": processed_count,
                    "total_api_calls": len(responses),
                    "new_api_calls": len(responses) - initial_api_calls,
                    "cache_hits": processed_count - (len(responses) - initial_api_calls),
                    "elapsed_seconds": elapsed_time,
                    "rows_per_second": processed_count / elapsed_time if elapsed_time > 0 else 0,
                },
                step=processed_count,
            )

        # Save checkpoint if save_interval is set and we've hit the interval
        if (
            proc_args.save_interval > 0
            and processed_count % proc_args.save_interval == 0
        ):
            logger.info(
                f"Checkpoint: Saving progress ({processed_count}/{len(rows_to_process)} processed)..."
            )
            save_dataframe(df, data_args.output_file)

    # Calculate final metrics
    total_time = time.time() - start_time
    total_api_calls = len(responses) - initial_api_calls
    cache_hit_rate = (processed_count - total_api_calls) / processed_count if processed_count > 0 else 0

    # Log final metrics
    tracker.log_metrics(
        {
            "total_rows_processed": processed_count,
            "total_api_calls": total_api_calls,
            "total_cache_hits": processed_count - total_api_calls,
            "cache_hit_rate": cache_hit_rate,
            "total_time_seconds": total_time,
            "avg_time_per_row": total_time / processed_count if processed_count > 0 else 0,
        }
    )

    # Final save
    logger.info(f"Saving final annotated data to {data_args.output_file}")
    save_dataframe(df, data_args.output_file)
    logger.info(f"Successfully saved {len(df)} annotated samples")
    logger.info(f"Total unique API calls made: {len(responses)} (rest were cached)")

    # Finish tracking
    tracker.finish()


if __name__ == "__main__":
    main()
