"""Tests for BasePipeline.initialize_client and initialize_tracker."""

import os
from typing import Any, Dict, List, Tuple, Type
from unittest.mock import MagicMock, patch

import pandas as pd
from openai import OpenAI

from syntk.config import (
    ConfigArguments,
    APIArguments,
    GenerationArguments,
    BaseDataArguments,
    BaseProcessingArguments,
)
from syntk.tracking import TrackingArguments, ExperimentTracker
from syntk.pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Minimal concrete pipeline for testing
# ---------------------------------------------------------------------------

class MinimalPipeline(BasePipeline):
    def __init__(self, api_key_env: str = "TEST_API_KEY", base_url: str = "http://localhost:8080/v1"):
        super().__init__()
        self.config_args = ConfigArguments()
        self.api_args = APIArguments(api_key_env=api_key_env, base_url=base_url)
        self.gen_args = GenerationArguments()
        self.data_args = BaseDataArguments()
        self.proc_args = BaseProcessingArguments()
        self.track_args = TrackingArguments()

    def get_argument_classes(self) -> Tuple[Type, ...]:
        return (
            ConfigArguments, APIArguments, GenerationArguments,
            BaseDataArguments, BaseProcessingArguments, TrackingArguments,
        )

    def setup_dataframe(self, df: pd.DataFrame, resuming: bool) -> pd.DataFrame:
        return df

    def process_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        return {}

    def get_config_params(self) -> Dict[str, Any]:
        return {"model": self.api_args.model}

    def get_rows_to_process(self, df: pd.DataFrame) -> List[int]:
        return list(range(len(df)))


# ---------------------------------------------------------------------------
# initialize_client
# ---------------------------------------------------------------------------

class TestInitializeClient:
    def test_client_is_set_after_init(self):
        """pipeline.client is an OpenAI instance after initialize_client()."""
        pipeline = MinimalPipeline()
        pipeline.initialize_client()
        assert isinstance(pipeline.client, OpenAI)

    def test_uses_api_key_from_env(self):
        """When env var is set, its value is used as the api_key."""
        pipeline = MinimalPipeline(api_key_env="MY_KEY_VAR")
        with patch.dict(os.environ, {"MY_KEY_VAR": "sk-real-key"}):
            pipeline.initialize_client()
        assert pipeline.client.api_key == "sk-real-key"

    def test_falls_back_to_placeholder_when_env_missing(self):
        """Missing env var → placeholder key (no raise)."""
        pipeline = MinimalPipeline(api_key_env="DEFINITELY_NOT_SET_XYZ")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEFINITELY_NOT_SET_XYZ", None)
            pipeline.initialize_client()
        assert pipeline.client.api_key == "placeholder-api-key"

    def test_warns_when_api_key_missing(self, caplog):
        """A WARNING is logged when the env var is absent."""
        import logging

        pipeline = MinimalPipeline(api_key_env="DEFINITELY_NOT_SET_XYZ")
        os.environ.pop("DEFINITELY_NOT_SET_XYZ", None)
        with caplog.at_level(logging.WARNING):
            pipeline.initialize_client()
        assert any("placeholder" in r.message.lower() or "not found" in r.message.lower()
                   for r in caplog.records)

    def test_base_url_applied_to_client(self):
        """Client is created with the configured base_url."""
        pipeline = MinimalPipeline(base_url="http://myserver:1234/v1")
        pipeline.initialize_client()
        assert "myserver" in pipeline.client.base_url.host


# ---------------------------------------------------------------------------
# initialize_tracker
# ---------------------------------------------------------------------------

class TestInitializeTracker:
    def test_tracker_is_set(self):
        """pipeline.tracker is an ExperimentTracker after initialize_tracker()."""
        pipeline = MinimalPipeline()
        pipeline.get_config_params = MagicMock(return_value={"model": "test"})
        pipeline.initialize_tracker()
        assert isinstance(pipeline.tracker, ExperimentTracker)

    def test_get_config_params_called(self):
        """initialize_tracker calls get_config_params to log config."""
        pipeline = MinimalPipeline()
        pipeline.get_config_params = MagicMock(return_value={"model": "my-model"})
        pipeline.initialize_tracker()
        pipeline.get_config_params.assert_called_once()

    def test_none_config_params_not_logged(self):
        """None values in config_params are filtered out before log_params."""
        pipeline = MinimalPipeline()
        pipeline.get_config_params = MagicMock(
            return_value={"model": "x", "optional_field": None}
        )
        with patch.object(ExperimentTracker, "log_params") as mock_log:
            pipeline.initialize_tracker()
        logged = mock_log.call_args[0][0]
        assert "optional_field" not in logged
        assert logged.get("model") == "x"
