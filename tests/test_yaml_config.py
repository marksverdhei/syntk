"""Tests for BasePipeline._merge_yaml_config YAML loading and CLI override merging."""

import yaml
import pytest
from pathlib import Path
from typing import Tuple, Type, Dict, Any

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


# ---------------------------------------------------------------------------
# Minimal concrete pipeline for testing
# ---------------------------------------------------------------------------

class MinimalPipeline(BasePipeline):
    """Minimal concrete subclass to test BasePipeline methods."""

    def __init__(self):
        super().__init__()
        # Set up default argument instances so _merge_yaml_config can run
        self.config_args = ConfigArguments()
        self.api_args = APIArguments()
        self.gen_args = GenerationArguments()
        self.data_args = BaseDataArguments()
        self.proc_args = BaseProcessingArguments()
        self.track_args = TrackingArguments()

    def get_argument_classes(self) -> Tuple[Type, ...]:
        return (
            ConfigArguments,
            APIArguments,
            GenerationArguments,
            BaseDataArguments,
            BaseProcessingArguments,
            TrackingArguments,
        )

    def setup_dataframe(self, df: pd.DataFrame, resuming: bool) -> pd.DataFrame:
        return df

    def process_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        return {}

    def get_config_params(self) -> Dict[str, Any]:
        return {}

    def get_rows_to_process(self, df: pd.DataFrame):
        return []


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


# ---------------------------------------------------------------------------
# _merge_yaml_config: loading from YAML
# ---------------------------------------------------------------------------

class TestMergeYamlConfig:
    def _arg_classes_without_config(self):
        return (
            APIArguments,
            GenerationArguments,
            BaseDataArguments,
            BaseProcessingArguments,
            TrackingArguments,
        )

    def test_yaml_model_loaded(self, tmp_path):
        """model field from YAML is applied when CLI is default."""
        cfg = _write_yaml(tmp_path, {"model": "custom-model"})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.api_args.model == "custom-model"

    def test_yaml_base_url_loaded(self, tmp_path):
        cfg = _write_yaml(tmp_path, {"base_url": "http://myserver:1234/v1"})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.api_args.base_url == "http://myserver:1234/v1"

    def test_yaml_temperature_loaded(self, tmp_path):
        cfg = _write_yaml(tmp_path, {"temperature": 0.7})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.gen_args.temperature == pytest.approx(0.7)

    def test_yaml_input_output_files_loaded(self, tmp_path):
        cfg = _write_yaml(
            tmp_path, {"input_file": "data.csv", "output_file": "out.parquet"}
        )
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.data_args.input_file == "data.csv"
        assert pipeline.data_args.output_file == "out.parquet"

    def test_yaml_log_interval_loaded(self, tmp_path):
        cfg = _write_yaml(tmp_path, {"log_interval": 25})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.proc_args.log_interval == 25

    def test_extra_yaml_keys_ignored(self, tmp_path):
        """Unknown YAML keys (allow_extra_keys=True) do not cause errors."""
        cfg = _write_yaml(
            tmp_path,
            {"model": "test-model", "unknown_key_xyz": "some_value"},
        )
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        # Should not raise
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.api_args.model == "test-model"


# ---------------------------------------------------------------------------
# CLI value takes priority over YAML when it differs from the default
# ---------------------------------------------------------------------------

class TestCliOverridesTakesPriority:
    def _arg_classes_without_config(self):
        return (
            APIArguments,
            GenerationArguments,
            BaseDataArguments,
            BaseProcessingArguments,
            TrackingArguments,
        )

    def test_cli_model_overrides_yaml(self, tmp_path):
        """If CLI model differs from default, it takes priority over YAML."""
        cfg = _write_yaml(tmp_path, {"model": "yaml-model"})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline.api_args.model = "cli-model"  # simulates --model cli-model
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.api_args.model == "cli-model"

    def test_yaml_used_when_cli_is_default(self, tmp_path):
        """When CLI value equals the dataclass default, YAML takes priority."""
        # Default model is "gpt-3.5-turbo" — leave it unchanged
        cfg = _write_yaml(tmp_path, {"model": "yaml-model"})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        # api_args.model is still the default
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.api_args.model == "yaml-model"

    def test_cli_temperature_overrides_yaml(self, tmp_path):
        cfg = _write_yaml(tmp_path, {"temperature": 0.5})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline.gen_args.temperature = 0.9  # CLI override
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.gen_args.temperature == pytest.approx(0.9)

    def test_multiple_cli_overrides_applied(self, tmp_path):
        cfg = _write_yaml(
            tmp_path, {"model": "yaml-model", "temperature": 0.5, "log_interval": 5}
        )
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline.api_args.model = "cli-model"
        pipeline.gen_args.temperature = 1.0
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        # CLI overrides win
        assert pipeline.api_args.model == "cli-model"
        assert pipeline.gen_args.temperature == pytest.approx(1.0)
        # YAML value used when CLI was default
        assert pipeline.proc_args.log_interval == 5

    def test_yaml_none_value_used_for_optional_field(self, tmp_path):
        """YAML setting an optional field to a value when CLI default is None."""
        cfg = _write_yaml(tmp_path, {"max_tokens": 512})
        pipeline = MinimalPipeline()
        pipeline.config_args.config_file = str(cfg)
        pipeline._merge_yaml_config(self._arg_classes_without_config())
        assert pipeline.gen_args.max_tokens == 512
