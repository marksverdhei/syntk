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
            logging_dir="/custom/logs"
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

    @patch("syntk.tracking.ExperimentTracker._init_tensorboard")
    def test_tensorboard_initialization(self, mock_init):
        """Test TensorBoard tracker initialization."""
        _tracker = ExperimentTracker(report_to=["tensorboard"], run_name="test_run")
        mock_init.assert_called_once()

    @patch("syntk.tracking.ExperimentTracker._init_mlflow")
    def test_mlflow_initialization(self, mock_init):
        """Test MLflow tracker initialization."""
        _tracker = ExperimentTracker(report_to=["mlflow"], run_name="test_run")
        mock_init.assert_called_once()

    @patch("syntk.tracking.ExperimentTracker._init_wandb")
    def test_wandb_initialization(self, mock_init):
        """Test W&B tracker initialization."""
        _tracker = ExperimentTracker(report_to=["wandb"], run_name="test_run")
        mock_init.assert_called_once()

    @patch("syntk.tracking.ExperimentTracker._init_aim")
    def test_aim_initialization(self, mock_init):
        """Test Aim tracker initialization."""
        _tracker = ExperimentTracker(report_to=["aim"], run_name="test_run")
        mock_init.assert_called_once()

    @patch("syntk.tracking.ExperimentTracker._init_tensorboard")
    @patch("syntk.tracking.ExperimentTracker._init_wandb")
    def test_multiple_trackers_initialization(self, mock_wandb, mock_tb):
        """Test initialization with multiple trackers."""
        _tracker = ExperimentTracker(
            report_to=["tensorboard", "wandb"],
            run_name="test_run"
        )
        mock_tb.assert_called_once()
        mock_wandb.assert_called_once()

    def test_log_params_with_mocked_tracker(self):
        """Test log_params with a mocked tracker."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()

        params = {"model": "gpt-4", "temperature": 0.7}
        tracker.log_params(params)

        tracker.tb_writer.add_text.assert_called_once()

    def test_log_metrics_with_mocked_tracker(self):
        """Test log_metrics with a mocked tracker."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()

        metrics = {"rows_processed": 100, "api_calls": 50}
        tracker.log_metrics(metrics, step=10)

        assert tracker.tb_writer.add_scalar.call_count == 2

    def test_finish_with_mocked_tracker(self):
        """Test finish with a mocked tracker."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()

        tracker.finish()

        tracker.tb_writer.close.assert_called_once()

    def test_log_params_handles_exceptions(self, caplog):
        """Test that log_params handles exceptions gracefully."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()
        tracker.tb_writer.add_text.side_effect = Exception("Test error")

        tracker.log_params({"param": "value"})

        assert "Failed to log params to tensorboard" in caplog.text

    def test_log_metrics_handles_exceptions(self, caplog):
        """Test that log_metrics handles exceptions gracefully."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()
        tracker.tb_writer.add_scalar.side_effect = Exception("Test error")

        tracker.log_metrics({"metric": 1.0})

        assert "Failed to log metrics to tensorboard" in caplog.text

    def test_finish_handles_exceptions(self, caplog):
        """Test that finish handles exceptions gracefully."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()
        tracker.tb_writer.close.side_effect = Exception("Test error")

        tracker.finish()

        assert "Failed to finish tensorboard" in caplog.text


class TestExperimentTrackerContextManager:
    """Tests for ExperimentTracker context manager."""

    def test_context_manager_calls_finish(self):
        """Test that context manager calls finish on exit."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()

        with tracker:
            pass

        tracker.tb_writer.close.assert_called_once()

    def test_context_manager_calls_finish_on_exception(self):
        """Test that context manager calls finish even on exception."""
        tracker = ExperimentTracker(report_to=None)
        tracker.trackers = ["tensorboard"]
        tracker.tb_writer = Mock()

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
        args = TrackingArguments(
            report_to="tensorboard, wandb",
            run_name="test"
        )

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        def mock_init_wandb(self):
            self.wandb = Mock()
            self.trackers.append("wandb")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb), \
             patch("syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb):
            tracker = get_tracker(args)
            assert len(tracker.trackers) == 2

    def test_get_tracker_handles_whitespace(self):
        """Test that get_tracker handles whitespace in report_to."""
        args = TrackingArguments(report_to="  tensorboard  ,  wandb  ")

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        def mock_init_wandb(self):
            self.wandb = Mock()
            self.trackers.append("wandb")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb), \
             patch("syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb):
            tracker = get_tracker(args)
            assert len(tracker.trackers) == 2

    def test_get_tracker_with_custom_logging_dir(self):
        """Test get_tracker with custom logging_dir."""
        args = TrackingArguments(
            report_to="tensorboard",
            logging_dir="/custom/path"
        )

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard"):
            tracker = get_tracker(args)
            assert tracker.logging_dir == "/custom/path"

    def test_get_tracker_filters_empty_strings(self):
        """Test that get_tracker filters out empty strings."""
        args = TrackingArguments(report_to="tensorboard,,wandb,")

        def mock_init_tb(self):
            self.tb_writer = Mock()
            self.trackers.append("tensorboard")

        def mock_init_wandb(self):
            self.wandb = Mock()
            self.trackers.append("wandb")

        with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb), \
             patch("syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb):
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
