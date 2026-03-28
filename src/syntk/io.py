"""File I/O utilities for dataframe operations."""

import os
import pandas as pd


def _ensure_hf_dataset_repo_exists(output_file: str) -> None:
    """Auto-create a private HuggingFace dataset repo if the path is hf:// and the repo doesn't exist.

    Args:
        output_file: Output file path, e.g. hf://datasets/owner/repo/file.tsv
    """
    # hf://datasets/<owner>/<repo>/<path>
    rest = output_file[len("hf://datasets/"):]
    parts = rest.split("/")
    if len(parts) < 2:
        return
    repo_id = f"{parts[0]}/{parts[1]}"
    try:
        from huggingface_hub import create_repo
        create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception:
        pass


def save_dataframe(df: pd.DataFrame, output_file: str) -> None:
    """Save dataframe to file, detecting format from extension.

    Supported formats: .parquet, .csv, .json, .jsonl, .tsv
    Creates output directories if needed.

    Args:
        df: DataFrame to save
        output_file: Output file path

    Raises:
        ValueError: If file format is not supported
    """
    if output_file.startswith("hf://datasets/"):
        _ensure_hf_dataset_repo_exists(output_file)
    else:
        # Ensure output directory exists for local paths
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
    """Load dataframe from file, detecting format from extension.

    Supported formats: .parquet, .csv, .json, .jsonl, .tsv

    Args:
        input_file: Input file path

    Returns:
        Loaded DataFrame

    Raises:
        ValueError: If file format is not supported
    """
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


def find_available_column_name(df: pd.DataFrame, base_name: str) -> str:
    """Find an available column name by appending numbers if necessary.

    Args:
        df: DataFrame to check for existing columns
        base_name: Desired column name

    Returns:
        base_name if available, otherwise base_name_N where N is first available number
    """
    if base_name not in df.columns:
        return base_name

    counter = 1
    while f"{base_name}_{counter}" in df.columns:
        counter += 1
    return f"{base_name}_{counter}"
