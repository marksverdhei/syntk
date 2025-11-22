# syntk

A synthetic data toolkit for generating and augmenting datasets using Large Language Models.

## Overview

`syntk` provides pipelines for generating synthetic data and enriching existing datasets with LLM-powered annotations. It supports various data formats (Parquet, CSV, JSON, JSONL, TSV) and any OpenAI-compatible API endpoint.

### Key Features

- **Column Pipeline**: Fill dataset columns with LLM-generated values
- **Flexible Data Formats**: Support for Parquet, CSV, JSON, JSONL, and TSV
- **OpenAI-Compatible APIs**: Works with OpenAI, OpenRouter, and any compatible endpoint
- **Resume Support**: Automatically resumes interrupted processing
- **Configurable**: YAML configuration files for reproducible pipelines
- **Experiment Tracking**: Built-in support for TensorBoard, MLflow, W&B, and Aim

## Installation

### Using pip

```bash
pip install git+https://github.com/marksverdhei/syntk.git
```

### Using uv (for development)

This project uses [`uv`](https://github.com/astral-sh/uv), which can be installed in one line:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and install the project:

```bash
git clone https://github.com/marksverdhei/syntk.git
cd syntk
uv sync --dev
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

### Column Pipeline

The column pipeline fills a dataset column with LLM-generated values based on a prompt template.

1. **Set up your API key** (using OpenRouter as an example):

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

2. **Run the example pipeline**:

```bash
syntk column examples/column_rate_difficulty.yaml
```

This example annotates Norwegian text reviews with difficulty scores for sentiment classification.

### Creating Your Own Pipeline

Create a YAML configuration file:

```yaml
# API configuration
base_url: https://openrouter.ai/api/v1
api_key_env: "OPENROUTER_API_KEY"
model: "anthropic/claude-sonnet-4.5"

# Input/output
input_file: "data/input.parquet"
output_file: "data/output.parquet"
output_column: "generated_column"
limit: null  # null for all, integer for count, float (0-1) for fraction

# Prompt template (can reference any column from the dataset)
prompt_template: |-
  Your prompt here. Reference columns like {column_name}.

save_interval: 100  # Save progress every N rows
```

Run your pipeline:

```bash
syntk column your_config.yaml
```

### Command-Line Options

Override any config file setting via command-line:

```bash
syntk column config.yaml --limit 100 --model "gpt-4" --output_file "custom_output.csv"
```

View all available options:

```bash
syntk column --help
```

## Experiment Tracking

`syntk` supports experiment tracking with TensorBoard, MLflow, Weights & Biases, and Aim to monitor generation metrics in real-time.

### Installation

Install tracking dependencies (choose what you need):

```bash
# TensorBoard
pip install syntk[tensorboard]

# MLflow
pip install syntk[mlflow]

# Weights & Biases
pip install syntk[wandb]

# Aim
pip install syntk[aim]

# All trackers
pip install syntk[tracking]
```

### Usage

Add tracking configuration to your YAML file:

```yaml
# Enable experiment tracking (comma-separated for multiple)
report_to: "tensorboard"  # Options: tensorboard, mlflow, wandb, aim
run_name: "my_experiment"
logging_dir: "./logs"  # Directory for logs or MLflow tracking URI
```

Or via command line:

```bash
syntk column config.yaml --report_to tensorboard --run_name my_experiment
```

### Tracked Metrics

For inference/generation pipelines, `syntk` tracks:

- **rows_processed**: Number of rows generated
- **total_api_calls**: LLM API calls made
- **cache_hits**: Responses served from cache
- **cache_hit_rate**: Cache efficiency percentage
- **rows_per_second**: Generation throughput
- **avg_time_per_row**: Average generation time

### Viewing Results

**TensorBoard:**
```bash
tensorboard --logdir ./logs/tensorboard
```

**MLflow:**
```bash
mlflow ui --backend-store-uri ./logs
```

**W&B:** Visit the URL printed at run start

**Aim:**
```bash
aim up --repo ./logs
```

## Available Pipelines

- **`column`**: Fill dataset columns with LLM-generated values

More pipelines coming soon!

## Development

Run tests:

```bash
pytest
```

Format and lint:

```bash
ruff check
ruff format
```
