import os
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from syntk.tracking import TrackingArguments, get_tracker
from transformers import HfArgumentParser

tqdm.pandas()

logger = logging.getLogger(__name__)


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
        default="Process the following text:\n\n{text}",
        metadata={
            "help": "Prompt template with placeholders for column names (e.g., {text}, {label})"
        },
    )
    log_interval: int = field(
        default=10,
        metadata={"help": "Log metrics every N rows (0 to only log at end)"},
    )
    save_interval: int = field(
        default=100,
        metadata={"help": "Save progress every N rows (0 to only save at end)"},
    )


def get_chat_response(
    client: OpenAI,
    prompt: str,
    api_args: APIArguments,
    gen_args: GenerationArguments,
    return_raw: bool = False,
) -> dict:
    """Get response from OpenAI-compatible API."""
    logger.debug(f"API Call - Model: {api_args.model}")
    logger.debug(
        f"API Call - Temperature: {gen_args.temperature}, Max tokens: {gen_args.max_tokens}"
    )
    logger.debug(
        f"API Call - Prompt: {prompt[:200]}..."
        if len(prompt) > 200
        else f"API Call - Prompt: {prompt}"
    )

    # Build kwargs dict, only including non-None values
    kwargs = {
        "model": api_args.model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if gen_args.temperature is not None:
        kwargs["temperature"] = gen_args.temperature
    if gen_args.max_tokens is not None:
        kwargs["max_tokens"] = gen_args.max_tokens
    if gen_args.top_p is not None:
        kwargs["top_p"] = gen_args.top_p
    if gen_args.frequency_penalty is not None:
        kwargs["frequency_penalty"] = gen_args.frequency_penalty
    if gen_args.presence_penalty is not None:
        kwargs["presence_penalty"] = gen_args.presence_penalty

    response = client.chat.completions.create(**kwargs)

    message = response.choices[0].message
    content = message.content
    reasoning_content = getattr(message, "reasoning_content", None)
    stop_reason = response.choices[0].finish_reason

    # Log warning if content is None
    if content is None:
        logger.warning(
            f"API response returned None content. Stop reason: {stop_reason}"
        )

    logger.debug(
        f"API Response: {content[:200]}..."
        if content and len(content) > 200
        else f"API Response: {content}"
    )
    logger.debug(
        f"API Call - Tokens used: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}"
    )

    result = {
        "content": content,
        "reasoning_content": reasoning_content,
        "stop_reason": stop_reason,
    }

    # Add raw request/response data if requested
    if return_raw:
        result["raw"] = {
            "request": kwargs,
            "response": response.model_dump()
            if hasattr(response, "model_dump")
            else response.dict(),
        }

    return result


def save_dataframe(df: pd.DataFrame, output_file: str) -> None:
    """Save dataframe to file, detecting format from extension."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_file_lower = output_file.lower()
    if output_file_lower.endswith(".parquet"):
        df.to_parquet(output_file, index=False)
    elif output_file_lower.endswith(".csv"):
        df.to_csv(output_file, index=False)
    elif output_file_lower.endswith(".json"):
        df.to_json(output_file, orient="records", lines=False)
    elif output_file_lower.endswith(".jsonl"):
        df.to_json(output_file, orient="records", lines=True)
    elif output_file_lower.endswith(".tsv"):
        df.to_csv(output_file, sep="\t", index=False)
    else:
        raise ValueError(
            f"Unsupported output format: {output_file}. Supported formats: .parquet, .csv, .json, .jsonl, .tsv"
        )


def find_available_column_name(df: pd.DataFrame, base_name: str) -> str:
    """Find an available column name by appending numbers if necessary."""
    if base_name not in df.columns:
        return base_name

    counter = 1
    while f"{base_name}_{counter}" in df.columns:
        counter += 1
    return f"{base_name}_{counter}"


def load_dataframe(input_file: str) -> pd.DataFrame:
    """Load dataframe from file, detecting format from extension."""
    input_file_lower = input_file.lower()
    if input_file_lower.endswith(".parquet"):
        return pd.read_parquet(input_file)
    elif input_file_lower.endswith(".csv"):
        return pd.read_csv(input_file)
    elif input_file_lower.endswith(".json") or input_file_lower.endswith(".jsonl"):
        return pd.read_json(input_file, lines=input_file_lower.endswith(".jsonl"))
    elif input_file_lower.endswith(".tsv"):
        return pd.read_csv(input_file, sep="\t")
    else:
        raise ValueError(
            f"Unsupported file format: {input_file}. Supported formats: .parquet, .csv, .json, .jsonl, .tsv"
        )


def save_raw_api_call(file_path: str, row_index: int, result: dict) -> None:
    """Append raw API request/response to JSONL file.

    Args:
        file_path: Path to JSONL file
        row_index: Index of the row being processed
        result: Result dict containing raw API data
    """
    import json
    import time

    if "raw" not in result:
        return

    record = {
        "timestamp": time.time(),
        "row_index": row_index,
        "request": result["raw"]["request"],
        "response": result["raw"]["response"],
    }

    # Append to JSONL file (one JSON object per line)
    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def process_row(
    row,
    client: OpenAI,
    api_args: APIArguments,
    gen_args: GenerationArguments,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
    responses: dict,
) -> dict:
    """Process a single row by generating text based on the prompt template.

    Returns:
        dict with keys: content, reasoning_content, stop_reason, and optionally raw
    """
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
    return_raw = data_args.raw_api_json_path is not None
    response = get_chat_response(
        client, prompt, api_args, gen_args, return_raw=return_raw
    )
    responses[prompt] = response
    return response


def main() -> None:
    """Main function to process dataset rows with LLM-generated content."""
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
            (
                APIArguments,
                GenerationArguments,
                DataArguments,
                ProcessingArguments,
                TrackingArguments,
            )
        )
        (
            api_args_yaml,
            gen_args_yaml,
            data_args_yaml,
            proc_args_yaml,
            track_args_yaml,
        ) = yaml_parser.parse_dict(config_dict, allow_extra_keys=True)

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
        "log_interval": proc_args.log_interval,
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

    # Initialize optional columns for reasoning content and stop reason
    actual_stop_reason_column = None
    if data_args.save_stop_reason:
        actual_stop_reason_column = find_available_column_name(df, "stop_reason")
        if not resuming or actual_stop_reason_column not in df.columns:
            df[actual_stop_reason_column] = pd.NA
        logger.info(
            f"Stop reasons will be saved to column: {actual_stop_reason_column}"
        )

    if data_args.reasoning_content_column:
        if not resuming or data_args.reasoning_content_column not in df.columns:
            df[data_args.reasoning_content_column] = pd.NA
        logger.info(
            f"Reasoning content will be saved to column: {data_args.reasoning_content_column}"
        )

    # Identify rows to process (those with missing values in output column)
    rows_to_process = df[df[data_args.output_column].isna()].index.tolist()

    if len(rows_to_process) == 0:
        logger.info("All rows already processed. Nothing to do.")
        return

    # Initialize raw API JSON file if specified (clear file if not resuming)
    if data_args.raw_api_json_path and not resuming:
        raw_dir = os.path.dirname(data_args.raw_api_json_path)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
        # Create empty file (or truncate if exists)
        open(data_args.raw_api_json_path, "w").close()
        logger.info(f"Raw API data will be saved to: {data_args.raw_api_json_path}")

    logger.info(f"Processing {len(rows_to_process)} rows...")

    # Track start time
    start_time = time.time()

    # Process rows one at a time with periodic checkpointing
    processed_count = 0
    initial_api_calls = len(responses)

    for idx in tqdm(rows_to_process, desc="Generating"):
        row = df.loc[idx]
        result = process_row(
            row, client, api_args, gen_args, data_args, proc_args, responses
        )

        # Save content to output column
        df.at[idx, data_args.output_column] = result["content"]

        # Save reasoning content if column is specified
        if data_args.reasoning_content_column:
            df.at[idx, data_args.reasoning_content_column] = result["reasoning_content"]

        # Save stop reason if enabled
        if actual_stop_reason_column:
            df.at[idx, actual_stop_reason_column] = result["stop_reason"]

        # Save raw API request/response if enabled
        if data_args.raw_api_json_path:
            save_raw_api_call(data_args.raw_api_json_path, idx, result)

        processed_count += 1

        # Log metrics periodically
        if proc_args.log_interval > 0 and processed_count % proc_args.log_interval == 0:
            elapsed_time = time.time() - start_time
            tracker.log_metrics(
                {
                    "rows_processed": processed_count,
                    "total_api_calls": len(responses),
                    "new_api_calls": len(responses) - initial_api_calls,
                    "cache_hits": processed_count
                    - (len(responses) - initial_api_calls),
                    "elapsed_seconds": elapsed_time,
                    "rows_per_second": processed_count / elapsed_time
                    if elapsed_time > 0
                    else 0,
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
    cache_hit_rate = (
        (processed_count - total_api_calls) / processed_count
        if processed_count > 0
        else 0
    )

    # Log final summary (single scalars, not time series)
    tracker.log_summary(
        {
            "total_rows_processed": processed_count,
            "total_api_calls": total_api_calls,
            "total_cache_hits": processed_count - total_api_calls,
            "cache_hit_rate": cache_hit_rate,
            "total_time_seconds": total_time,
            "avg_time_per_row": total_time / processed_count
            if processed_count > 0
            else 0,
        }
    )

    # Final save
    logger.info(f"Saving final processed data to {data_args.output_file}")
    save_dataframe(df, data_args.output_file)
    logger.info(f"Successfully saved {len(df)} processed samples")
    logger.info(f"Total unique API calls made: {len(responses)} (rest were cached)")

    # Finish tracking
    tracker.finish()


if __name__ == "__main__":
    main()
