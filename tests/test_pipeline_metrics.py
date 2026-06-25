"""Tests for BasePipeline._log_metrics and _log_final_summary."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from syntk.pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Minimal subclass for instantiation
# ---------------------------------------------------------------------------

class _Pipeline(BasePipeline):
    def get_argument_classes(self):
        return ()

    def setup_dataframe(self, df, resuming):
        return df

    def process_row(self, row, idx):
        return {}

    def get_config_params(self):
        return {}

    def get_rows_to_process(self, df):
        return []


def _make_pipeline(responses=None, initial_api_calls=0):
    p = _Pipeline()
    p.tracker = MagicMock()
    p.responses = responses if responses is not None else {}
    return p


# ---------------------------------------------------------------------------
# _log_metrics
# ---------------------------------------------------------------------------

class TestLogMetrics:
    def test_calls_tracker_log_metrics(self):
        p = _make_pipeline(responses={"r1": "a", "r2": "b"})
        start = time.time() - 5.0  # 5 seconds ago
        p._log_metrics(processed_count=10, initial_api_calls=0, start_time=start)
        p.tracker.log_metrics.assert_called_once()

    def test_rows_processed_is_processed_count(self):
        p = _make_pipeline(responses={})
        start = time.time() - 1.0
        p._log_metrics(processed_count=7, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_metrics.call_args[0][0]
        assert kwargs["rows_processed"] == 7

    def test_total_api_calls_matches_responses(self):
        p = _make_pipeline(responses={"a": 1, "b": 2, "c": 3})
        start = time.time() - 1.0
        p._log_metrics(processed_count=3, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_metrics.call_args[0][0]
        assert kwargs["total_api_calls"] == 3

    def test_new_api_calls_delta(self):
        """new_api_calls = current - initial_api_calls."""
        p = _make_pipeline(responses={"a": 1, "b": 2, "c": 3})
        start = time.time() - 1.0
        p._log_metrics(processed_count=3, initial_api_calls=1, start_time=start)
        kwargs = p.tracker.log_metrics.call_args[0][0]
        assert kwargs["new_api_calls"] == 2

    def test_cache_hits(self):
        """cache_hits = processed_count - new_api_calls."""
        p = _make_pipeline(responses={"a": 1})
        start = time.time() - 1.0
        # 5 rows processed, 1 api call (started at 0) → 4 cache hits
        p._log_metrics(processed_count=5, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_metrics.call_args[0][0]
        assert kwargs["cache_hits"] == 4

    def test_rows_per_second_positive_elapsed(self):
        p = _make_pipeline()
        start = time.time() - 10.0  # 10 seconds elapsed
        p._log_metrics(processed_count=100, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_metrics.call_args[0][0]
        assert kwargs["rows_per_second"] > 0

    def test_rows_per_second_zero_elapsed(self):
        """Zero elapsed time returns 0 rows_per_second (no division by zero)."""
        p = _make_pipeline()
        # Use a future start time to force elapsed ≈ 0
        start = time.time() + 1000
        p._log_metrics(processed_count=5, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_metrics.call_args[0][0]
        assert kwargs["rows_per_second"] == 0

    def test_step_is_processed_count(self):
        p = _make_pipeline()
        start = time.time() - 1.0
        p._log_metrics(processed_count=42, initial_api_calls=0, start_time=start)
        _, call_kwargs = p.tracker.log_metrics.call_args
        assert call_kwargs.get("step") == 42


# ---------------------------------------------------------------------------
# _log_final_summary
# ---------------------------------------------------------------------------

class TestLogFinalSummary:
    def test_calls_tracker_log_summary(self):
        p = _make_pipeline(responses={"r1": "a"})
        start = time.time() - 5.0
        p._log_final_summary(processed_count=10, initial_api_calls=0, start_time=start)
        p.tracker.log_summary.assert_called_once()

    def test_total_rows_processed(self):
        p = _make_pipeline(responses={})
        start = time.time() - 1.0
        p._log_final_summary(processed_count=8, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_summary.call_args[0][0]
        assert kwargs["total_rows_processed"] == 8

    def test_cache_hit_rate_all_cache(self):
        """If all rows were cache hits (0 new API calls), rate should be 1.0."""
        p = _make_pipeline(responses={})
        start = time.time() - 1.0
        # 10 rows processed, 0 new api calls → rate = 1.0
        p._log_final_summary(processed_count=10, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_summary.call_args[0][0]
        assert kwargs["cache_hit_rate"] == pytest.approx(1.0)

    def test_cache_hit_rate_no_cache(self):
        """If all rows were API calls, rate should be 0.0."""
        p = _make_pipeline(responses={"a": 1, "b": 2, "c": 3})
        start = time.time() - 1.0
        # 3 rows processed, 3 new api calls → rate = 0.0
        p._log_final_summary(processed_count=3, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_summary.call_args[0][0]
        assert kwargs["cache_hit_rate"] == pytest.approx(0.0)

    def test_cache_hit_rate_zero_processed(self):
        """Zero rows processed returns cache_hit_rate=0 (no division by zero)."""
        p = _make_pipeline(responses={})
        start = time.time() - 1.0
        p._log_final_summary(processed_count=0, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_summary.call_args[0][0]
        assert kwargs["cache_hit_rate"] == 0

    def test_avg_time_per_row_zero_processed(self):
        """Zero rows → avg_time_per_row is 0 (no division by zero)."""
        p = _make_pipeline(responses={})
        start = time.time() - 1.0
        p._log_final_summary(processed_count=0, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_summary.call_args[0][0]
        assert kwargs["avg_time_per_row"] == 0

    def test_total_time_is_positive(self):
        p = _make_pipeline()
        start = time.time() - 5.0
        p._log_final_summary(processed_count=3, initial_api_calls=0, start_time=start)
        kwargs = p.tracker.log_summary.call_args[0][0]
        assert kwargs["total_time_seconds"] > 0
