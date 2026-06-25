"""Tests for ExperimentTracker.log_summary and trackio backend."""

from unittest.mock import Mock, patch
from syntk.tracking import ExperimentTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tb_tracker():
    """Return a tracker with a mocked TensorBoard backend."""
    def mock_init_tb(self):
        self.tb_writer = Mock()
        self.trackers.append("tensorboard")

    with patch("syntk.tracking.ExperimentTracker._init_tensorboard", mock_init_tb):
        return ExperimentTracker(report_to=["tensorboard"])


# ---------------------------------------------------------------------------
# log_summary — TensorBoard
# ---------------------------------------------------------------------------

class TestLogSummaryTensorboard:
    def test_log_summary_writes_text(self):
        tracker = _make_tb_tracker()
        tracker.log_summary({"accuracy": 0.95, "f1": 0.88})
        tracker.tb_writer.add_text.assert_called_once()
        tag, text = tracker.tb_writer.add_text.call_args[0]
        assert tag == "summary"
        assert "accuracy: 0.95" in text
        assert "f1: 0.88" in text

    def test_log_summary_no_trackers_noop(self):
        tracker = ExperimentTracker(report_to=None)
        tracker.log_summary({"accuracy": 0.9})  # must not raise

    def test_log_summary_handles_exception(self, caplog):
        tracker = _make_tb_tracker()
        tracker.tb_writer.add_text.side_effect = RuntimeError("disk full")
        tracker.log_summary({"k": 1.0})  # must not raise
        assert "Failed to log summary to tensorboard" in caplog.text


# ---------------------------------------------------------------------------
# log_summary — MLflow
# ---------------------------------------------------------------------------

class TestLogSummaryMlflow:
    def test_log_summary_calls_log_metrics(self):
        def mock_init_mlflow(self):
            self.mlflow = Mock()
            self.trackers.append("mlflow")

        with patch(
            "syntk.tracking.ExperimentTracker._init_mlflow", mock_init_mlflow
        ):
            tracker = ExperimentTracker(report_to=["mlflow"])
            tracker.log_summary({"precision": 0.7})
            tracker.mlflow.log_metrics.assert_called_once_with({"precision": 0.7})


# ---------------------------------------------------------------------------
# log_summary — W&B
# ---------------------------------------------------------------------------

class TestLogSummaryWandb:
    def test_log_summary_sets_run_summary(self):
        def mock_init_wandb(self, use_trackio=False):
            mock_run = Mock()
            mock_run.summary = {}
            mock_wandb = Mock()
            mock_wandb.run = mock_run
            self.wandb = mock_wandb
            self.trackers.append("wandb")

        with patch(
            "syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb
        ):
            tracker = ExperimentTracker(report_to=["wandb"])
            tracker.log_summary({"recall": 0.82})
            assert tracker.wandb.run.summary["recall"] == 0.82


# ---------------------------------------------------------------------------
# log_summary — Trackio backend
# ---------------------------------------------------------------------------

class TestLogSummaryTrackio:
    def test_log_summary_calls_trackio_log(self):
        def mock_init_wandb(self, use_trackio=False):
            mock_trackio = Mock()
            self.trackio = mock_trackio
            self.trackers.append("trackio")

        with patch(
            "syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb
        ):
            tracker = ExperimentTracker(report_to=["trackio"])
            tracker.log_summary({"f1": 0.91})
            tracker.trackio.log.assert_called_once_with({"f1": 0.91})


# ---------------------------------------------------------------------------
# log_summary — Aim backend
# ---------------------------------------------------------------------------

class TestLogSummaryAim:
    def test_log_summary_calls_aim_set(self):
        def mock_init_aim(self):
            self.aim_run = Mock()
            self.trackers.append("aim")

        with patch(
            "syntk.tracking.ExperimentTracker._init_aim", mock_init_aim
        ):
            tracker = ExperimentTracker(report_to=["aim"])
            tracker.log_summary({"score": 0.66})
            tracker.aim_run.set.assert_called_once_with(
                ("summary", "score"), 0.66, strict=False
            )


# ---------------------------------------------------------------------------
# Trackio — init and log_metrics
# ---------------------------------------------------------------------------

class TestTrackioBackend:
    def test_trackio_init_sets_trackio_attr(self):
        def mock_init_wandb(self, use_trackio=False):
            mock_trackio = Mock()
            self.trackio = mock_trackio
            self.trackers.append("trackio")

        with patch(
            "syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb
        ):
            tracker = ExperimentTracker(report_to=["trackio"])
            assert "trackio" in tracker.trackers
            assert hasattr(tracker, "trackio")

    def test_trackio_log_metrics_calls_log(self):
        def mock_init_wandb(self, use_trackio=False):
            self.trackio = Mock()
            self.trackers.append("trackio")

        with patch(
            "syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb
        ):
            tracker = ExperimentTracker(report_to=["trackio"])
            tracker.log_metrics({"loss": 0.3}, step=5)
            tracker.trackio.log.assert_called_once_with({"loss": 0.3}, step=5)

    def test_trackio_log_params_calls_config_update(self):
        def mock_init_wandb(self, use_trackio=False):
            self.trackio = Mock()
            self.trackers.append("trackio")

        with patch(
            "syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb
        ):
            tracker = ExperimentTracker(report_to=["trackio"])
            tracker.log_params({"lr": 1e-4})
            tracker.trackio.config.update.assert_called_once_with({"lr": 1e-4})

    def test_trackio_finish_calls_finish(self):
        def mock_init_wandb(self, use_trackio=False):
            self.trackio = Mock()
            self.trackers.append("trackio")

        with patch(
            "syntk.tracking.ExperimentTracker._init_wandb", mock_init_wandb
        ):
            tracker = ExperimentTracker(report_to=["trackio"])
            tracker.finish()
            tracker.trackio.finish.assert_called_once()

    def test_trackio_missing_dependency(self, caplog):
        with patch(
            "syntk.tracking.ExperimentTracker._init_wandb",
            side_effect=ImportError("No module named 'trackio'"),
        ):
            tracker = ExperimentTracker(report_to=["trackio"])
            assert "Could not initialize trackio" in caplog.text
            assert tracker.trackers == []
