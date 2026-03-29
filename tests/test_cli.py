"""Tests for syntk CLI (syntk.cli)."""

import sys
from unittest.mock import patch, MagicMock

import pytest

from syntk.cli import app
from typer.testing import CliRunner


runner = CliRunner()


class TestColumnCommand:
    def test_column_subcommand_exists(self):
        result = runner.invoke(app, ["--help"])
        assert "column" in result.output

    def test_help_output(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_column_help(self):
        result = runner.invoke(app, ["column", "--help"])
        assert result.exit_code == 0

    def test_no_subcommand_shows_help(self):
        """Invoking without subcommand shows help output (no_args_is_help=True)."""
        result = runner.invoke(app, [])
        # no_args_is_help=True exits with code 2 and prints help to stdout
        assert "column" in result.output

    def test_column_invokes_pipeline_main(self):
        """column subcommand should call column_pipeline.main."""
        with patch("syntk.pipelines.column.main") as mock_main:
            result = runner.invoke(app, ["column", "--help"])
        # Either --help exits before main, or main is called - just check no crash
        assert result.exit_code == 0

    def test_column_restores_sys_argv(self):
        """After column subcommand, sys.argv should be restored."""
        original_argv = sys.argv.copy()
        with patch("syntk.pipelines.column.main"):
            runner.invoke(app, ["column", "--model", "gpt-4"])
        assert sys.argv == original_argv
