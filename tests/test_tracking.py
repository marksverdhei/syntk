"""Tests for experiment tracking functionality."""

import pytest
from unittest.mock import Mock, patch
from syntk.tracking import (
    TrackingArguments,
    ExperimentTracker,
    get_tracker,
)


class TestTrackingArguments:
    """Tests for TrackingArguments dataclass."""

    def test_default_values(self):
        """Test that TrackingArguments has correct defaults."""
        args = TrackingArguments()
        assert args.report_to is None
        assert args.run_name is None
        assert args.logging_dir == "./logs"

    def test_custom_values(self):
        """Test TrackingArguments with custom values."""
        args = TrackingArguments(
            report_to="tensorboard,wandb",
            run_name="test_run",
            logging_dir="/custom/logs",
        )
        assert args.report_to == "tensorboard,wandb"
        assert args.run_name == "test_run"
        assert args.logging_dir == "/custom/logs"


class TestExperimentTrackerNoTrackers:
    """Tests for ExperimentTracker when no trackers are specified."""

    def test_initialization_with_none(self):
        """Test that tracker initializes correctly with no trackers."""
        tracker = ExperimentTracker(report_to=None)
        assert tracker.trackers == []
        assert tracker.run_name == "syntk_run"

    def test_initialization_with_empty_list(self):
        """Test that tracker initializes correctly with empty list."""
        tracker = ExperimentTracker(report_to=[])
        assert tracker.trackers == []

    def test_log_params_no_op(self):
        """Test that log_params does nothing when no trackers."""
        tracker = ExperimentTracker(report_to=None)
        tracker.log_params({"param1": "value1"})  # Should not raise

    def test_log_metrics_no_op(self):
        """Test that log_metrics does nothing when no trackers."""
        tracker = ExperimentTracker(report_to=None)
        tracker.log_metrics({"metric1": 1.0}, step=0)  # Should not raise

    def test_finish_no_op(self):
        """Test that finish does nothing when no trackers."""
        tracker = ExperimentTracker(report_to=None)
        tracker.finish()  # Should not raise

    def test_context_manager_no_trackers(self):
        """Test context manager works with no trackers."""
        with ExperimentTracker(report_to=None) as tracker:
            assert tracker.trackers == []


class TestExperimentTrackerUnknownTracker:
    """Tests for ExperimentTracker with unknown tracker names."""

    def test_unknown_tracker_logs_warning(self, caplog):
        """Test that unknown tracker names log a warning."""
        tracker = ExperimentTracker(report_to=["unknown_tracker"])
        assert "Unknown tracker: unknown_tracker" in caplog.text
        assert tracker.trackers == []


class TestExperimentTrackerMissingDependencies:
    """Tests for ExperimentTracker when tracking libraries are not installed."""

    def test_tensorboard_missing_dependency(self, caplog):
        """Test graceful handling when tensorboard is not installed."""
        with patch("syntk.tracking.ExperimentTracker._init_tensorboard") as mock_init:
            mock_init.side_effect = ImportError("No module named 'torch'")
            tracker = ExperimentTracker(report_to=["tensorboard"])

            assert "Could not initialize tensorboard" in caplog.text
            assert tracker.trackers == []

    def test_mlflow_missing_dependency(self, caplog):
        """Test graceful handling when mlflow is not installed."""
        with patch("syntk.tracking.ExperimentTracker._init_mlflow") as mock_init:
            mock_init.side_effect = ImportError("No module named 'mlflow'")
            tracker = ExperimentTracker(report_to=["mlflow"])

            assert "Could not initialize mlflow" in caplog.text
            assert tracker.trackers == []

    def test_wandb_missing_dependency(self, caplog):
        """Test graceful handling when wandb is not installed."""
        with patch("syntk.tracking.ExperimentTracker._init_wandb") as mock_init:
            mock_init.side_effect = ImportError("No module named 'wandb'")
            tracker = ExperimentTracker(report_to=["wandb"])

            assert "Could not initialize wandb" in caplog.text
            assert tracker.trackers == []

    def test_aim_missing_dependency(self, caplog):
        """Test graceful handling when aim is not installed."""
        with patch("syntk.tracking.ExperimentTracker._init_aim") as mock_init:
            mock_init.side_effect = ImportError("No module named 'aim'")
            tracker = ExperimentTracker(report_to=["aim"])

            assert "Could not initialize aim" in caplog.text
            assert tracker.trackers == []


class TestExperimentTrackerWithMocks:
    """Tests for ExperimentTracker with mocked tracking libraries."""

    def test_log_params_dispatches_correctly(self):
        """Test that log_params calls correct methods and passes correct data."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])
            params = {"model": "gpt-4", "temperature": 0.7}
            tracker.log_params(params)

            # Verify correct method called with correct data
            tracker.tb_writer.add_text.assert_called_once()
            call_args = tracker.tb_writer.add_text.call_args
            assert call_args[0][0] == "config"
            # Verify actual content contains the params
            assert "model: gpt-4" in call_args[0][1]
            assert "temperature: 0.7" in call_args[0][1]

    def test_log_metrics_dispatches_correctly(self):
        """Test that log_metrics calls correct methods for each metric."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])
            metrics = {"rows_processed": 100, "api_calls": 50}
            tracker.log_metrics(metrics, step=10)

            # Verify add_scalar called for each metric
            assert tracker.tb_writer.add_scalar.call_count == 2
            # Verify correct arguments passed
            calls = tracker.tb_writer.add_scalar.call_args_list
            assert calls[0][0] == ("rows_processed", 100, 10)
            assert calls[1][0] == ("api_calls", 50, 10)

    def test_finish_calls_cleanup(self):
        """Test that finish calls cleanup methods on all trackers."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])
            tracker.finish()

            tracker.tb_writer.close.assert_called_once()

    def test_log_params_handles_exceptions(self, caplog):
        """Test that log_params handles exceptions gracefully."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.tb_writer.add_text.side_effect = Exception("Test error")
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])
            tracker.log_params({"param": "value"})  # Should not raise

            assert "Failed to log params to tensorboard" in caplog.text

    def test_log_metrics_handles_exceptions(self, caplog):
        """Test that log_metrics handles exceptions gracefully."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.tb_writer.add_scalar.side_effect = Exception("Test error")
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])
            tracker.log_metrics({"metric": 1.0})  # Should not raise

            assert "Failed to log metrics to tensorboard" in caplog.text

    def test_finish_handles_exceptions(self, caplog):
        """Test that finish handles exceptions gracefully."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.tb_writer.close.side_effect = Exception("Test error")
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])
            tracker.finish()  # Should not raise

            assert "Failed to finish tensorboard" in caplog.text


class TestExperimentTrackerContextManager:
    """Tests for ExperimentTracker context manager."""

    def test_context_manager_calls_finish(self):
        """Test that context manager calls finish on exit."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])

            with tracker:
                pass

            tracker.tb_writer.close.assert_called_once()

    def test_context_manager_calls_finish_on_exception(self):
        """Test that context manager calls finish even on exception."""

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
            tracker = ExperimentTracker(report_to=["tensorboard"])

            with pytest.raises(ValueError):
                with tracker:
                    raise ValueError("Test error")

            tracker.tb_writer.close.assert_called_once()


class TestGetTracker:
    """Tests for get_tracker factory function."""

    def test_get_tracker_with_none(self):
        """Test get_tracker with report_to=None."""
        args = TrackingArguments(report_to=None)
        tracker = get_tracker(args)

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.trackers == []

    def test_get_tracker_with_single_tracker(self):
        """Test get_tracker with a single tracker."""
        args = TrackingArguments(report_to="tensorboard", run_name="test")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard"):
            tracker = get_tracker(args)
            assert tracker.run_name == "test"

    def test_get_tracker_with_multiple_trackers(self):
        """Test get_tracker with comma-separated trackers."""
        args = TrackingArguments(report_to="tensorboard, wandb", run_name="test")

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        def mock_init_wandb(self, use_trackio=False):
            setattr(self, "trackio" if use_trackio else "wandb", Mock())
            self.trackers.append("trackio" if use_trackio else "wandb")

        with (
            patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb),
            patch("syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb),
        ):
            tracker = get_tracker(args)
            assert len(tracker.trackers) == 2

    def test_get_tracker_handles_whitespace(self):
        """Test that get_tracker handles whitespace in report_to."""
        args = TrackingArguments(report_to="  tensorboard  ,  wandb  ")

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        def mock_init_wandb(self, use_trackio=False):
            setattr(self, "trackio" if use_trackio else "wandb", Mock())
            self.trackers.append("trackio" if use_trackio else "wandb")

        with (
            patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb),
            patch("syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb),
        ):
            tracker = get_tracker(args)
            assert len(tracker.trackers) == 2

    def test_get_tracker_with_custom_logging_dir(self):
        """Test get_tracker with custom logging_dir."""
        args = TrackingArguments(report_to="tensorboard", logging_dir="/custom/path")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard"):
            tracker = get_tracker(args)
            assert tracker.logging_dir == "/custom/path"

    def test_get_tracker_filters_empty_strings(self):
        """Test that get_tracker filters out empty strings."""
        args = TrackingArguments(report_to="tensorboard,,wandb,")

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        def mock_init_wandb(self, use_trackio=False):
            setattr(self, "trackio" if use_trackio else "wandb", Mock())
            self.trackers.append("trackio" if use_trackio else "wandb")

        with (
            patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb),
            patch("syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb),
        ):
            tracker = get_tracker(args)
            # Should only have 2 trackers, not 4
            assert len(tracker.trackers) == 2


class TestTrackerNameNormalization:
    """Tests for tracker name normalization."""

    @patch("syntk.tracking.ExperimentTracker._init_tensorboard")
    def test_tensorboard_case_insensitive(self, mock_init):
        """Test that tracker names are case-insensitive."""
        for name in ["tensorboard", "TensorBoard", "TENSORBOARD"]:
            _tracker = ExperimentTracker(report_to=[name])
            mock_init.assert_called()
            mock_init.reset_mock()

    @patch("syntk.tracking.ExperimentTracker._init_mlflow")
    def test_mlflow_case_insensitive(self, mock_init):
        """Test MLflow name is case-insensitive."""
        for name in ["mlflow", "MLflow", "MLFLOW"]:
            _tracker = ExperimentTracker(report_to=[name])
            mock_init.assert_called()
            mock_init.reset_mock()
