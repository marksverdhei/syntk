"""File I/O utilities for dataframe operations."""

import os
import logging
import pandas as pd
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def _parse_hf_url(url: str) -> tuple[str, str]:
    """Parse a Hugging Face URL to extract repo ID and file path.

    Args:
        url: Hugging Face URL in format hf://datasets/username/repo-name/path/to/file.ext

    Returns:
        Tuple of (repo_id, file_path)

    Example:
        >>> _parse_hf_url("hf://datasets/user/repo/file.tsv")
        ('user/repo', 'file.tsv')
    """
    # Remove hf:// prefix
    path = url.replace("hf://", "")
    # Split into parts: ['datasets', 'username', 'repo-name', 'path', 'to', 'file.ext']
    parts = path.split("/", 3)
    if len(parts) < 4:
        raise ValueError(f"Invalid Hugging Face URL format: {url}")

    # repo_type is parts[0] (e.g., 'datasets' or 'models')
    # repo_id is parts[1]/parts[2] (e.g., 'username/repo-name')
    # file_path is parts[3] (e.g., 'path/to/file.ext')
    repo_id = f"{parts[1]}/{parts[2]}"
    file_path = parts[3]
    return repo_id, file_path


def _ensure_hf_repo_exists(output_file: str) -> None:
    """Ensure Hugging Face repository exists, creating it if necessary.

    Args:
        output_file: Output file path (should be a hf:// URL)
    """
    if not output_file.startswith("hf://"):
        return

    try:
        repo_id, _ = _parse_hf_url(output_file)

        # Determine repo type from URL
        repo_type = "dataset" if "/datasets/" in output_file else None

        # Try to create the repository
        api = HfApi()
        try:
            api.create_repo(
                repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True
            )
            logger.info(f"Ensured Hugging Face repository exists: {repo_id}")
        except Exception as e:
            # Log at debug level since the repo might exist but we don't have access to verify
            logger.debug(
                f"Could not ensure repository exists (it may already exist): {e}"
            )
    except Exception as e:
        logger.warning(f"Failed to parse or check Hugging Face URL: {e}")


def save_dataframe(df: pd.DataFrame, output_file: str) -> None:
    """Save dataframe to file, detecting format from extension.

    Supported formats: .parquet, .csv, .json, .jsonl, .tsv
    Creates output directories if needed.
    For Hugging Face URLs (hf://datasets/...), automatically creates private
    dataset repositories if they don't exist.

    Args:
        df: DataFrame to save
        output_file: Output file path or Hugging Face URL

    Raises:
        ValueError: If file format is not supported
    """
    # For HF URLs, ensure the repository exists
    if output_file.startswith("hf://"):
        _ensure_hf_repo_exists(output_file)
    else:
        # Ensure output directory exists for local files
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
