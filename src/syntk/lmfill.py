import os
import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from transformers import HfArgumentParser

tqdm.pandas()

logger = logging.getLogger(__name__)

responses = {}


@dataclass
class ConfigArguments:
    """Arguments for configuration file."""
    config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML config file. If provided, loads defaults from file."}
    )


@dataclass
class APIArguments:
    """Arguments for API configuration."""
    base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL for OpenAI-compatible API"}
    )
    api_key_env: str = field(
        default="OPENAI_API_KEY",
        metadata={"help": "Environment variable name for API key"}
    )
    model: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "Model name to use"}
    )


@dataclass
class GenerationArguments:
    """Arguments for text generation."""
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Sampling temperature"}
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum tokens to generate"}
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={"help": "Nucleus sampling probability"}
    )
    frequency_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "Frequency penalty"}
    )
    presence_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "Presence penalty"}
    )


@dataclass
class DataArguments:
    """Arguments for data input/output."""
    input_file: str = field(
        default="hf://datasets/ltg/norec_sentence/ternary/train-00000-of-00001.parquet",
        metadata={"help": "Input parquet file path"}
    )
    output_file: str = field(
        default="annotated_difficulty.parquet",
        metadata={"help": "Output parquet file path"}
    )
    text_column: str = field(
        default="review",
        metadata={"help": "Name of the text column in the dataset"}
    )
    output_column: str = field(
        default="difficulty_annotation",
        metadata={"help": "Name of the generated output column"}
    )
    limit: Optional[float] = field(
        default=None,
        metadata={"help": "Limit samples: integer for count, 0-1 for fraction, None for all"}
    )


@dataclass
class ProcessingArguments:
    """Arguments for processing configuration."""
    prompt_template: str = field(
        default="""Analyze the difficulty of classifying the sentiment of the following Norwegian text:

Text: {text}
True sentiment: {sentiment}

Rate the difficulty on a scale of 1-5 and provide a brief explanation.""",
        metadata={"help": "Prompt template with {text} and {sentiment} placeholders"}
    )
    save_interval: int = field(
        default=100,
        metadata={"help": "Save progress every N rows (0 to only save at end)"}
    )


def get_chat_response(
    client: OpenAI,
    prompt: str,
    api_args: APIArguments,
    gen_args: GenerationArguments
) -> str:
    """Get response from OpenAI-compatible API."""
    logger.debug(f"API Call - Model: {api_args.model}")
    logger.debug(f"API Call - Temperature: {gen_args.temperature}, Max tokens: {gen_args.max_tokens}")
    logger.debug(f"API Call - Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"API Call - Prompt: {prompt}")

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

    response_text = response.choices[0].message.content
    logger.debug(f"API Response: {response_text[:200]}..." if len(response_text) > 200 else f"API Response: {response_text}")
    logger.debug(f"API Call - Tokens used: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}")

    return response_text


def save_dataframe(df: pd.DataFrame, output_file: str) -> None:
    """Save dataframe to file, detecting format from extension."""
    output_file_lower = output_file.lower()
    if output_file_lower.endswith('.parquet'):
        df.to_parquet(output_file, index=False)
    elif output_file_lower.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file_lower.endswith('.json'):
        df.to_json(output_file, orient='records', lines=False)
    elif output_file_lower.endswith('.jsonl'):
        df.to_json(output_file, orient='records', lines=True)
    elif output_file_lower.endswith('.tsv'):
        df.to_csv(output_file, sep='\t', index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_file}. Supported formats: .parquet, .csv, .json, .jsonl, .tsv")


def load_dataframe(input_file: str) -> pd.DataFrame:
    """Load dataframe from file, detecting format from extension."""
    input_file_lower = input_file.lower()
    if input_file_lower.endswith('.parquet'):
        return pd.read_parquet(input_file)
    elif input_file_lower.endswith('.csv'):
        return pd.read_csv(input_file)
    elif input_file_lower.endswith('.json') or input_file_lower.endswith('.jsonl'):
        return pd.read_json(input_file, lines=input_file_lower.endswith('.jsonl'))
    elif input_file_lower.endswith('.tsv'):
        return pd.read_csv(input_file, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {input_file}. Supported formats: .parquet, .csv, .json, .jsonl, .tsv")


def annotate_difficulty(
    row,
    client: OpenAI,
    api_args: APIArguments,
    gen_args: GenerationArguments,
    data_args: DataArguments,
    proc_args: ProcessingArguments
) -> str:
    """Annotate a single row with difficulty rating."""
    # Create a dictionary of all column values for formatting
    # This allows any column to be referenced in the prompt template
    format_dict = row.to_dict()

    # Format the prompt using all available columns
    prompt = proc_args.prompt_template.format(**format_dict)

    # Use cache if available
    if prompt in responses:
        logger.debug(f"Using cached response for prompt")
        return responses[prompt]

    logger.debug(f"Making new API call (cache miss)")
    response = get_chat_response(client, prompt, api_args, gen_args)
    responses[prompt] = response
    return response


def main() -> None:
    """Main function to annotate difficulty ratings."""
    import sys
    import yaml

    # Check if first positional argument is a YAML config file
    config_file = None
    args_to_parse = None
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml')) and not sys.argv[1].startswith('--'):
        config_file = sys.argv[1]
        args_to_parse = sys.argv[2:]  # Parse remaining args after the config file

    # Parse command line arguments
    parser = HfArgumentParser((ConfigArguments, APIArguments, GenerationArguments, DataArguments, ProcessingArguments))
    config_args, api_args, gen_args, data_args, proc_args = parser.parse_args_into_dataclasses(args=args_to_parse)

    # Set config file if it was provided as positional argument
    if config_file:
        config_args.config_file = config_file

    # If config file is specified, load it and merge with CLI args
    if config_args.config_file:
        with open(config_args.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse from flattened YAML dict
        yaml_parser = HfArgumentParser((APIArguments, GenerationArguments, DataArguments, ProcessingArguments))
        api_args_yaml, gen_args_yaml, data_args_yaml, proc_args_yaml = yaml_parser.parse_dict(
            config_dict, allow_extra_keys=True
        )

        # For each argument, use CLI value if it differs from default, otherwise use YAML value
        for args_obj, yaml_obj in [(api_args, api_args_yaml), (gen_args, gen_args_yaml),
                                     (data_args, data_args_yaml), (proc_args, proc_args_yaml)]:
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
        raise ValueError(f"API key not found in environment variable: {api_args.api_key_env}")

    logger.info(f"Initializing OpenAI client with base_url: {api_args.base_url}, model: {api_args.model}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=api_args.base_url,
        api_key=api_key,
    )

    # Check if we can resume from existing output file
    resuming = False
    if os.path.exists(data_args.output_file):
        try:
            logger.info(f"Output file exists, checking for resume possibility: {data_args.output_file}")
            df = load_dataframe(data_args.output_file)

            if data_args.output_column in df.columns:
                # Count how many rows are already processed
                processed_count = df[data_args.output_column].notna().sum()
                total_count = len(df)
                logger.info(f"Found {processed_count}/{total_count} rows already processed. Resuming...")
                resuming = True
            else:
                logger.info(f"Output column '{data_args.output_column}' not found in output file. Loading input file instead.")
                df = load_dataframe(data_args.input_file)
        except Exception as e:
            logger.warning(f"Could not load output file for resume: {e}. Loading input file instead.")
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
            raise ValueError("Limit must be positive (integer for count, 0-1 for fraction)")

    # Initialize output column if not resuming
    if not resuming or data_args.output_column not in df.columns:
        df[data_args.output_column] = pd.NA

    # Identify rows to process (those with missing values in output column)
    rows_to_process = df[df[data_args.output_column].isna()].index.tolist()

    if len(rows_to_process) == 0:
        logger.info("All rows already processed. Nothing to do.")
        return

    logger.info(f"Processing {len(rows_to_process)} rows...")

    # Process rows one at a time with periodic checkpointing
    processed_count = 0
    for idx in tqdm(rows_to_process, desc="Generating"):
        row = df.loc[idx]
        result = annotate_difficulty(row, client, api_args, gen_args, data_args, proc_args)
        df.at[idx, data_args.output_column] = result
        processed_count += 1

        # Save checkpoint if save_interval is set and we've hit the interval
        if proc_args.save_interval > 0 and processed_count % proc_args.save_interval == 0:
            logger.info(f"Checkpoint: Saving progress ({processed_count}/{len(rows_to_process)} processed)...")
            save_dataframe(df, data_args.output_file)

    # Final save
    logger.info(f"Saving final annotated data to {data_args.output_file}")
    save_dataframe(df, data_args.output_file)
    logger.info(f"Successfully saved {len(df)} annotated samples")
    logger.info(f"Total unique API calls made: {len(responses)} (rest were cached)")


if __name__ == '__main__':
    main()
