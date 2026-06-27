"""Bootstrap pipeline — generate N new rows for a tabular dataset using LLM few-shot prompting."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import trange
from transformers import HfArgumentParser

from syntk.api_client import get_chat_response
from syntk.config import (
    APIArguments,
    ConfigArguments,
    GenerationArguments,
)
from syntk.io import load_dataframe, save_dataframe

logger = logging.getLogger(__name__)


@dataclass
class BootstrapDataArguments:
    """Data arguments for bootstrap pipeline."""

    input_file: str = field(
        default="input.parquet",
        metadata={"help": "Input file path to bootstrap from (.parquet, .csv, .json, .jsonl, .tsv)"},
    )
    output_file: str = field(
        default="output.parquet",
        metadata={"help": "Output file path for bootstrapped dataset"},
    )
    n: int = field(
        default=10,
        metadata={"help": "Number of new rows to generate"},
    )
    n_shots: int = field(
        default=3,
        metadata={"help": "Number of existing rows to show as few-shot examples per generation"},
    )
    columns: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of column names to include in generation. "
                "Defaults to all columns."
            )
        },
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed for reproducible few-shot sampling"},
    )
    system_prompt: str = field(
        default=(
            "You are a synthetic data generator. "
            "Given examples of rows from a dataset, generate a new row that follows the same style, "
            "format, and distribution as the examples. "
            "Respond with a single JSON object — no extra text, no markdown fences."
        ),
        metadata={"help": "System prompt sent to the model"},
    )
    append: bool = field(
        default=True,
        metadata={
            "help": (
                "If True (default), append generated rows to the input dataset and save to output_file. "
                "If False, save only the generated rows to output_file."
            )
        },
    )


def _rows_to_json_block(rows: List[Dict[str, Any]]) -> str:
    """Format a list of row dicts as a numbered JSON block for the prompt."""
    lines = []
    for i, row in enumerate(rows, 1):
        lines.append(f"Example {i}:")
        lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines)


def _build_user_message(examples: List[Dict[str, Any]], columns: List[str]) -> str:
    return (
        f"Here are {len(examples)} example rows from the dataset:\n\n"
        + _rows_to_json_block(examples)
        + f"\n\nGenerate one new row with the same columns: {columns}. "
        "Respond with a single JSON object only."
    )


def _parse_row_response(response: str, expected_columns: List[str]) -> Optional[Dict[str, Any]]:
    """Parse LLM response as JSON and validate expected columns are present."""
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        # Fill missing columns with None
        for col in expected_columns:
            obj.setdefault(col, None)
        # Drop unexpected columns
        return {col: obj[col] for col in expected_columns}
    except (json.JSONDecodeError, KeyError):
        return None


def bootstrap(
    client,
    df: pd.DataFrame,
    data_args: BootstrapDataArguments,
    api_args: APIArguments,
    gen_args: GenerationArguments,
) -> pd.DataFrame:
    """Generate N new rows for df and return the full bootstrapped dataframe.

    Args:
        client: OpenAI-compatible client
        df: Source dataframe to sample examples from
        data_args: Bootstrap data arguments
        api_args: API arguments
        gen_args: Generation arguments

    Returns:
        DataFrame containing only the newly generated rows.
    """
    if data_args.seed is not None:
        random.seed(data_args.seed)

    columns: List[str] = (
        [c.strip() for c in data_args.columns.split(",")]
        if data_args.columns
        else list(df.columns)
    )

    existing_rows = df[columns].to_dict(orient="records")
    new_rows: List[Dict[str, Any]] = []
    failed = 0

    gen_kwargs: Dict[str, Any] = {}
    if gen_args.temperature is not None:
        gen_kwargs["temperature"] = gen_args.temperature
    if gen_args.max_tokens is not None:
        gen_kwargs["max_tokens"] = gen_args.max_tokens
    if gen_args.top_p is not None:
        gen_kwargs["top_p"] = gen_args.top_p
    if gen_args.frequency_penalty is not None:
        gen_kwargs["frequency_penalty"] = gen_args.frequency_penalty
    if gen_args.presence_penalty is not None:
        gen_kwargs["presence_penalty"] = gen_args.presence_penalty

    for _ in trange(data_args.n, desc="Bootstrapping"):
        k = min(data_args.n_shots, len(existing_rows))
        examples = random.sample(existing_rows, k)

        user_msg = _build_user_message(examples, columns)
        messages = [
            {"role": "system", "content": data_args.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = get_chat_response(
            client=client,
            messages=messages,
            model=api_args.model,
            **gen_kwargs,
        )

        if response is None:
            logger.warning("API returned None response, skipping row")
            failed += 1
            continue

        parsed = _parse_row_response(response, columns)
        if parsed is None:
            logger.warning(f"Could not parse LLM response as JSON row: {response[:200]!r}")
            failed += 1
            continue

        new_rows.append(parsed)
        existing_rows.append(parsed)  # Include generated rows as potential future shots

    logger.info(f"Generated {len(new_rows)} rows ({failed} failed)")
    return pd.DataFrame(new_rows, columns=columns)


def main() -> None:
    """Entry point for syntk bootstrap."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Support positional YAML config
    args_to_parse = None
    if (
        len(sys.argv) > 1
        and sys.argv[1].endswith((".yaml", ".yml"))
        and not sys.argv[1].startswith("--")
    ):
        args_to_parse = sys.argv[2:]

    parser = HfArgumentParser(
        (ConfigArguments, APIArguments, GenerationArguments, BootstrapDataArguments)
    )
    config_args, api_args, gen_args, data_args = parser.parse_args_into_dataclasses(
        args=args_to_parse
    )

    # Initialize client
    api_key = os.getenv(api_args.api_key_env)
    if not api_key:
        api_key = "placeholder-api-key"
        logger.warning(
            f"API key not found in {api_args.api_key_env!r}. Using placeholder (local APIs only)."
        )

    from openai import OpenAI
    client = OpenAI(base_url=api_args.base_url, api_key=api_key)

    logger.info(f"Loading source data from {data_args.input_file}")
    df = load_dataframe(data_args.input_file)
    logger.info(f"Loaded {len(df)} rows — generating {data_args.n} new rows")

    generated_df = bootstrap(client, df, data_args, api_args, gen_args)

    if data_args.append:
        result_df = pd.concat([df, generated_df], ignore_index=True)
        logger.info(f"Appending to source: {len(df)} + {len(generated_df)} = {len(result_df)} rows")
    else:
        result_df = generated_df
        logger.info(f"Saving {len(result_df)} generated rows only")

    save_dataframe(result_df, data_args.output_file)
    logger.info(f"Saved to {data_args.output_file}")
