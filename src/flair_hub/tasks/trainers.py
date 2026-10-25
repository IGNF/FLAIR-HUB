import os
import sys
import time
import math
import torch.nn as nn

from pathlib import Path
from typing import Dict, Any
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from flair_hub.writer.prediction_writer import PredictionWriter
from flair_hub.tasks.recovery import auto_resume_path, finalize_recovery


STEP_PROGRESS_INTERVAL = 500


class OptimizerStepProgress(Callback):
    """Concise progress output for iteration-based training."""

    def __init__(
        self,
        total_steps: int,
        every_steps: int,
        monitor: str,
        checkpoint_callback: ModelCheckpoint,
    ) -> None:
        self.total_steps = int(total_steps)
        self.every_steps = max(1, int(every_steps))
        self.monitor = str(monitor)
        self.checkpoint_callback = checkpoint_callback
        self._last_reported = -1
        self._started_at = None
        self._started_step = 0

    def on_train_start(self, trainer, pl_module) -> None:
        """Report immediately, at step zero or at the restored resume step."""
        if trainer.global_rank != 0:
            return
        step = int(trainer.global_step)
        self._started_at = time.perf_counter()
        self._started_step = step
        self._print_progress(trainer, pl_module, step)
        self._last_reported = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if trainer.global_rank != 0 or step == self._last_reported:
            return
        if step != self.total_steps and step % self.every_steps:
            return
        self._last_reported = step
        self._print_progress(trainer, pl_module, step)

    def _print_progress(self, trainer, pl_module, step: int) -> None:
        datamodule = trainer.datamodule
        dataset = getattr(datamodule, "train_dataset", None)
        dataset_size = len(dataset) if dataset is not None else 0
        world_size = int(getattr(trainer, "world_size", 1))
        micro_batch = int(pl_module.config["hyperparams"]["batch_size"])
        effective_batch = micro_batch * world_size
        dataset_views = (
            step * effective_batch / dataset_size if dataset_size else float("nan")
        )
        metrics = trainer.callback_metrics
        train_loss = self._format_metric(metrics.get("train_loss"))
        train_miou = self._format_metric(metrics.get("train_miou"))
        current_monitor = self._format_metric(metrics.get(self.monitor))
        best_monitor = self._format_metric(self._best_monitor_value())

        if self.monitor == "train_loss":
            train_loss = f"{train_loss} (best {best_monitor})"
            monitor_text = None
        elif self.monitor == "train_miou":
            train_miou = f"{train_miou} (best {best_monitor})"
            monitor_text = None
        else:
            monitor_text = (
                f"{self._human_metric_name(self.monitor)} "
                f"{current_monitor} (best {best_monitor})"
            )

        metrics_parts = [
            f"train: loss {train_loss}, mIoU {train_miou}",
        ]
        if monitor_text is not None:
            metrics_parts.append(monitor_text)
        metrics_text = " | ".join(metrics_parts)

        elapsed_seconds = (
            max(0.0, time.perf_counter() - self._started_at)
            if self._started_at is not None
            else 0.0
        )
        completed_steps = max(0, step - self._started_step)
        if elapsed_seconds > 0.0 and completed_steps > 0:
            steps_per_second = completed_steps / elapsed_seconds
            remaining_steps = max(0, self.total_steps - step)
            eta_text = self._format_duration(
                remaining_steps / steps_per_second
            )
            speed_text = f"{steps_per_second:.2f} step/s"
        else:
            eta_text = "--:--:--"
            speed_text = "-- step/s"

        print(
            f"[TRAIN] opt.step {step:,}/{self.total_steps:,} "
            f"({100 * step / self.total_steps:.1f}%) | "
            f"views {dataset_views:.2f} | "
            f"{metrics_text} | "
            f"elapsed {self._format_duration(elapsed_seconds)} | "
            f"ETA {eta_text} | {speed_text}",
            flush=True,
        )

    def _best_monitor_value(self):
        """Return the finite best checkpoint score, including after resume."""
        value = self.checkpoint_callback.best_model_score
        if value is None:
            return None
        try:
            scalar = float(value)
        except (TypeError, ValueError, RuntimeError):
            return None
        return scalar if math.isfinite(scalar) else None

    @staticmethod
    def _format_metric(value: Any) -> str:
        """Format an optional scalar metric for the fixed progress layout."""
        if value is None:
            return "--.--"
        try:
            return f"{float(value):.5f}"
        except (TypeError, ValueError, RuntimeError):
            return str(value)

    @staticmethod
    def _human_metric_name(name: str) -> str:
        """Make configured metric keys compact and readable in progress logs."""
        return str(name).replace("_", " ").replace("miou", "mIoU")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a non-negative duration with unbounded hours."""
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def check_batchnorm_and_batch_size(config: Dict[str, Any], seg_module: nn.Module) -> None:
    """
    Check if the model contains BatchNorm layers and if the batch size is 1.
    If both conditions are met, print a message and abort the script.
    Args:
        config (dict): Configuration dictionary containing parameters for training.
        seg_module (nn.Module): Segmentation module for training.
    """
    batch_size = config['hyperparams']['batch_size']

    for module in seg_module.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and batch_size == 1:
            print("Warning: The model contains BatchNorm layers and the batch size is set to 1.")
            print("Aborting training to avoid potential issues.")
            print("Please set a batch size >1 in the current model provider configuration.")
            sys.exit(1)


def train(config: Dict[str, Any], data_module: Any, seg_module: nn.Module, out_dir: str) -> ModelCheckpoint:
    """
    Trains a model using the provided data module and segmentation module.
    Args:
        config (dict): Configuration dictionary containing parameters for training.
        data_module: Data module for training, validation, and testing.
        seg_module (nn.Module): Segmentation module for training.
        out_dir (str): Output directory for saving model checkpoints.
    Returns:
        ModelCheckpoint: Callback for model checkpointing.
    """
    check_batchnorm_and_batch_size(config, seg_module)

    duration_unit = config['hyperparams'].get('duration_unit', 'epochs')
    checkpoint_filename = (
        "ckpt-step{step:06d}-{val_loss:.2f}_"
        if duration_unit == 'steps'
        else "ckpt-epoch{epoch:02d}-{val_loss:.2f}_"
    ) + config['paths']["out_model_name"]
    ckpt_callback = ModelCheckpoint(
        monitor=config['saving']['ckpt_monitor'],
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename=checkpoint_filename,
        save_top_k=1,
        mode=config['saving']['ckpt_monitor_mode'],
        save_last=config['saving']['ckpt_save_also_last'],
        verbose=config['saving']['ckpt_verbose'],
        save_weights_only=config['saving']['ckpt_weights_only'],
    )

    callbacks = [ckpt_callback]
    progress_enabled = config['saving']["enable_progress_bar"]
    if duration_unit == 'steps' and progress_enabled:
        callbacks.append(
            OptimizerStepProgress(
                total_steps=config['hyperparams']['max_steps'],
                every_steps=STEP_PROGRESS_INTERVAL,
                monitor=config['saving']['ckpt_monitor'],
                checkpoint_callback=ckpt_callback,
            )
        )

    recovery_config = config['saving'].get('recovery_checkpoint', {})
    if recovery_config.get('enabled', False):
        recovery_interval = int(
            recovery_config.get('every_n_optimizer_steps', 2000)
        )
        if recovery_interval <= 0:
            raise ValueError(
                "saving.recovery_checkpoint.every_n_optimizer_steps must "
                "be strictly positive"
            )

        recovery_dir = os.path.join(
            out_dir,
            recovery_config.get('dir_name', 'recovery_checkpoints'),
        )
        recovery_callback = ModelCheckpoint(
            dirpath=recovery_dir,
            save_top_k=0,
            save_last=True,
            save_weights_only=False,
            every_n_train_steps=recovery_interval,
            save_on_train_epoch_end=False,
            verbose=config['saving']['ckpt_verbose'],
        )
        callbacks.append(recovery_callback)
        print(
            "Recovery checkpoint enabled: "
            f"{os.path.join(recovery_dir, 'last.ckpt')} every "
            f"{recovery_interval} optimizer steps"
        )

    if config['saving'].get('early_stopping', True):
        callbacks.append(
            EarlyStopping(
                monitor=config['saving']['ckpt_monitor'],
                min_delta=0.00,
                patience=config['saving']['ckpt_earlystopping_patience'],
                mode=config['saving']['ckpt_monitor_mode'],
            )
        )

    logger = TensorBoardLogger(
        save_dir=out_dir,
        name=Path(
            "tensorboard_logs_" + config['paths']["out_model_name"]
        ).as_posix(),
        version=0,
    )

    loggers = [logger]

    trainer_kwargs = dict(
        accelerator=config['hardware']["accelerator"],
        precision=config['hardware'].get("precision", "32-true"),
        devices=config['hardware']["gpus_per_node"],
        strategy=config['hardware']["strategy"],
        num_nodes=config['hardware']["num_nodes"],
        max_epochs=config['hyperparams']["num_epochs"],
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=(
            progress_enabled
            and duration_unit != 'steps'
        ),
    )
    max_steps = config['hyperparams'].get('max_steps')
    if max_steps is not None:
        trainer_kwargs['max_steps'] = int(max_steps)

    validation_interval = config['hyperparams'].get('validation_every')
    if duration_unit == 'steps':
        trainer_kwargs['val_check_interval'] = int(validation_interval)
        trainer_kwargs['check_val_every_n_epoch'] = None
    else:
        trainer_kwargs['check_val_every_n_epoch'] = int(validation_interval)

    trainer = Trainer(**trainer_kwargs)

    resume_mode = config['tasks']['train_tasks']['resume_training_from_ckpt']
    resume_path = config['paths'].get('ckpt_model_path')
    if isinstance(resume_mode, str):
        normalized_resume_mode = resume_mode.strip().lower()
        if normalized_resume_mode == 'auto':
            resume_path = auto_resume_path(resume_path, out_dir)
            resume_requested = bool(
                resume_path and os.path.isfile(resume_path)
            )
            if not resume_requested:
                print(
                    "No recovery or last checkpoint found; starting a fresh "
                    "training run."
                )
        elif normalized_resume_mode in {'true', 'yes', '1'}:
            resume_requested = True
        elif normalized_resume_mode in {'false', 'no', '0'}:
            resume_requested = False
        else:
            raise ValueError(
                "resume_training_from_ckpt must be true, false or 'auto'"
            )
    else:
        resume_requested = bool(resume_mode)

    if resume_requested:
        print('---------------------------------------------------------------')
        print('------------- RESUMING TRAINING FROM CKPT_PATH ----------------')
        print('---------------------------------------------------------------')
        if not resume_path or not os.path.isfile(resume_path):
            raise FileNotFoundError(
                "Checkpoint resume was requested but no checkpoint exists at "
                f"{resume_path!r}"
            )

        trainer.fit(
            seg_module,
            datamodule=data_module,
            ckpt_path=resume_path,
        )
    else:
        trainer.fit(seg_module, datamodule=data_module)

    if not ckpt_callback.best_model_path:

        (Path(out_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(Path(out_dir) / "checkpoints" / "last.ckpt"))
    finalize_recovery(trainer, config, out_dir)
    trainer.validate(seg_module, datamodule=data_module)

    return ckpt_callback


def predict(config: Dict[str, Any], data_module: Any, seg_module: nn.Module, out_dir: str) -> None:
    """
    This function makes predictions using the provided data module and segmentation module.
    Args:
        config (dict): Configuration dictionary containing parameters for prediction.
        data_module: Data module for training, validation, and testing.
        seg_module (nn.Module): Segmentation module for prediction.
        out_dir (str): Output directory for saving the predictions.
    """
    writer_callback = PredictionWriter(
        config,
        output_dir=os.path.join(out_dir),
        write_interval="batch",
    )

    trainer = Trainer(
        accelerator=config['hardware']["accelerator"],
        precision=config['hardware'].get("precision", "32-true"),
        devices=config['hardware']["gpus_per_node"],
        strategy=config['hardware']["strategy"],
        num_nodes=config['hardware']["num_nodes"],
        callbacks=[writer_callback],
        enable_progress_bar=config['saving']["enable_progress_bar"],
        logger=False,
    )

    trainer.predict(seg_module, datamodule=data_module, return_predictions=False)
