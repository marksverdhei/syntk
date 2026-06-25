"""Tests for BasePipeline.apply_limit — fraction, count, and validation."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from syntk.pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Minimal concrete subclass — only apply_limit needs testing here
# ---------------------------------------------------------------------------

class _Pipeline(BasePipeline):
    """Bare-minimum subclass so BasePipeline can be instantiated."""

    def get_argument_classes(self):
        return ()

    def setup_dataframe(self, df, resuming):
        return df

    def process_row(self, row):
        return {}

    def get_config_params(self):
        return {}

    def get_rows_to_process(self, df):
        return []


def _make_pipeline(limit=None) -> _Pipeline:
    p = _Pipeline()
    p.data_args = SimpleNamespace(limit=limit)
    return p


def _df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"x": range(n)})


# ---------------------------------------------------------------------------
# limit=None — no truncation
# ---------------------------------------------------------------------------

class TestApplyLimitNone:
    def test_none_returns_full_dataframe(self):
        pipeline = _make_pipeline(limit=None)
        df = _df(10)
        result = pipeline.apply_limit(df)
        assert len(result) == 10

    def test_none_is_identity(self):
        pipeline = _make_pipeline(limit=None)
        df = _df(5)
        result = pipeline.apply_limit(df)
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# Integer count (limit >= 1)
# ---------------------------------------------------------------------------

class TestApplyLimitCount:
    def test_integer_truncates_to_count(self):
        pipeline = _make_pipeline(limit=3)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 3

    def test_integer_larger_than_df_keeps_all(self):
        pipeline = _make_pipeline(limit=100)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 10

    def test_integer_exact_size(self):
        pipeline = _make_pipeline(limit=10)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 10

    def test_integer_one_keeps_one_row(self):
        pipeline = _make_pipeline(limit=1)
        result = pipeline.apply_limit(_df(5))
        assert len(result) == 1

    def test_float_above_one_treated_as_count(self):
        """limit=2.7 should be treated as count=2 (floor via int())."""
        pipeline = _make_pipeline(limit=2.7)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Fraction (0 < limit < 1)
# ---------------------------------------------------------------------------

class TestApplyLimitFraction:
    def test_half_truncates_to_half(self):
        pipeline = _make_pipeline(limit=0.5)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 5

    def test_fraction_rounds_down(self):
        """0.3 * 10 = 3.0 → 3 rows."""
        pipeline = _make_pipeline(limit=0.3)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 3

    def test_small_fraction_returns_at_least_zero(self):
        """Very small fraction may round to 0 rows — head(0) returns empty df."""
        pipeline = _make_pipeline(limit=0.01)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 0

    def test_fraction_close_to_one(self):
        """0.9 * 10 = 9 rows."""
        pipeline = _make_pipeline(limit=0.9)
        result = pipeline.apply_limit(_df(10))
        assert len(result) == 9


# ---------------------------------------------------------------------------
# Invalid limits
# ---------------------------------------------------------------------------

class TestApplyLimitInvalid:
    def test_zero_raises(self):
        pipeline = _make_pipeline(limit=0)
        with pytest.raises(ValueError, match="Limit must be positive"):
            pipeline.apply_limit(_df(10))

    def test_negative_raises(self):
        pipeline = _make_pipeline(limit=-1)
        with pytest.raises(ValueError, match="Limit must be positive"):
            pipeline.apply_limit(_df(10))

    def test_negative_fraction_raises(self):
        pipeline = _make_pipeline(limit=-0.5)
        with pytest.raises(ValueError, match="Limit must be positive"):
            pipeline.apply_limit(_df(10))
