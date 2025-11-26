"""Pipelines for synthetic data generation."""

from syntk.pipelines.base import BasePipeline
from syntk.pipelines.column import ColumnPipeline

__all__ = ["BasePipeline", "ColumnPipeline"]
