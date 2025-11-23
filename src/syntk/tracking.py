"""Experiment tracking integrations for syntk.

Simplified tracking integration inspired by TRL and LLaMA-Factory.
Supports TensorBoard, MLflow, W&B, Trackio, and Aim through optional dependencies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackingArguments:
    """Arguments for experiment tracking configuration.

    Similar to Transformers' TrainingArguments.report_to pattern.
    """

    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": "Experiment trackers to use (comma-separated): tensorboard, mlflow, wandb, aim, trackio. None to disable."
        },
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Name for the experiment run"}
    )
    logging_dir: Optional[str] = field(
        default="./logs",
        metadata={"help": "Directory for logs (tensorboard, aim) or MLflow tracking URI"},
    )


class ExperimentTracker:
    """Unified experiment tracker that manages multiple tracking backends.

    Inspired by HuggingFace's approach - one simple interface for all trackers.
    """

    def __init__(self, report_to: Optional[List[str]] = None, run_name: Optional[str] = None, logging_dir: str = "./logs"):
        """Initialize trackers based on report_to list."""
        self.trackers = []
        self.run_name = run_name or "syntk_run"
        self.logging_dir = logging_dir

        if not report_to:
            return

        for tracker_name in report_to:
            tracker_name = tracker_name.strip().lower()
            if not tracker_name:
                continue
            try:
                if tracker_name == "tensorboard":
                    self._init_tensorboard()
                elif tracker_name == "mlflow":
                    self._init_mlflow()
                elif tracker_name in ("wandb", "trackio"):
                    # trackio is wandb API-compatible, use same code path
                    self._init_wandb(use_trackio=(tracker_name == "trackio"))
                elif tracker_name == "aim":
                    self._init_aim()
                else:
                    logger.warning(f"Unknown tracker: {tracker_name}")
            except ImportError as e:
                logger.warning(f"Could not initialize {tracker_name}: {e}")

    def _init_tensorboard(self):
        """Initialize TensorBoard tracker."""
        import os

        # Try tensorboardX first (standalone), fall back to torch.utils.tensorboard
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            from torch.utils.tensorboard import SummaryWriter

        log_dir = os.path.join(self.logging_dir, "tensorboard", self.run_name)
        self.tb_writer = SummaryWriter(log_dir)
        self.trackers.append("tensorboard")
        logger.info(f"TensorBoard logging to: {log_dir}")

    def _init_mlflow(self):
        """Initialize MLflow tracker."""
        import mlflow

        mlflow.set_tracking_uri(self.logging_dir)
        mlflow.set_experiment("syntk")
        mlflow.start_run(run_name=self.run_name)
        self.mlflow = mlflow
        self.trackers.append("mlflow")
        logger.info(f"MLflow run started: {mlflow.active_run().info.run_id}")

    def _init_wandb(self, use_trackio=False):
        """Initialize W&B or Trackio tracker (trackio is wandb API-compatible)."""
        if use_trackio:
            import trackio as wandb
            tracker_name = "trackio"
        else:
            import wandb
            tracker_name = "wandb"

        wandb.init(project="syntk", name=self.run_name)
        self.wandb = wandb
        self.trackers.append(tracker_name)

        # Get run info (trackio doesn't have .url attribute)
        if wandb.run and hasattr(wandb.run, 'url'):
            logger.info(f"{tracker_name.capitalize()} run: {wandb.run.url}")
        else:
            logger.info(f"{tracker_name.capitalize()} run initialized: {self.run_name}")

    def _init_aim(self):
        """Initialize Aim tracker."""
        from aim import Run

        self.aim_run = Run(repo=self.logging_dir, experiment=self.run_name)
        self.trackers.append("aim")
        logger.info(f"Aim tracking to: {self.logging_dir}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to all active trackers."""
        if not self.trackers:
            return

        for tracker in self.trackers:
            try:
                if tracker == "tensorboard":
                    params_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
                    self.tb_writer.add_text("config", params_text)
                elif tracker == "mlflow":
                    self.mlflow.log_params(params)
                elif tracker in ("wandb", "trackio"):
                    self.wandb.config.update(params)
                elif tracker == "aim":
                    for k, v in params.items():
                        self.aim_run.set(k, v, strict=False)
            except Exception as e:
                logger.warning(f"Failed to log params to {tracker}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to all active trackers."""
        if not self.trackers:
            return

        for tracker in self.trackers:
            try:
                if tracker == "tensorboard":
                    for k, v in metrics.items():
                        self.tb_writer.add_scalar(k, v, step or 0)
                elif tracker == "mlflow":
                    self.mlflow.log_metrics(metrics, step=step)
                elif tracker in ("wandb", "trackio"):
                    self.wandb.log(metrics, step=step)
                elif tracker == "aim":
                    for k, v in metrics.items():
                        self.aim_run.track(v, name=k, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to {tracker}: {e}")

    def finish(self) -> None:
        """Finish all tracking runs."""
        for tracker in self.trackers:
            try:
                if tracker == "tensorboard":
                    self.tb_writer.close()
                elif tracker == "mlflow":
                    self.mlflow.end_run()
                elif tracker in ("wandb", "trackio"):
                    self.wandb.finish()
                elif tracker == "aim":
                    self.aim_run.close()
            except Exception as e:
                logger.warning(f"Failed to finish {tracker}: {e}")

        if self.trackers:
            logger.info(f"Finished tracking with: {', '.join(self.trackers)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def get_tracker(args: TrackingArguments) -> ExperimentTracker:
    """Create tracker from TrackingArguments.

    Args:
        args: TrackingArguments with report_to, run_name, and logging_dir

    Returns:
        ExperimentTracker instance
    """
    report_to_list = None
    if args.report_to:
        report_to_list = [t.strip() for t in args.report_to.split(",") if t.strip()]

    return ExperimentTracker(
        report_to=report_to_list,
        run_name=args.run_name,
        logging_dir=args.logging_dir,
    )
