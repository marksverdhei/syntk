"""Data I/O utilities for syntk.

Provides functions for loading and saving dataframes in various formats.
"""

import os

import pandas as pd


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
