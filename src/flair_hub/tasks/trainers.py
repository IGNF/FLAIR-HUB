import os
import sys
import torch
import torch.nn as nn

from pathlib import Path
from typing import Dict, Any
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from flair_hub.writer.prediction_writer import PredictionWriter


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

    ckpt_callback = ModelCheckpoint(
        monitor=config['saving']['ckpt_monitor'],
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="ckpt-{epoch:02d}-{val_loss:.2f}_" + config['paths']["out_model_name"],
        save_top_k=1,
        mode=config['saving']['ckpt_monitor_mode'],
        save_last=config['saving']['ckpt_save_also_last'],
        verbose=config['saving']['ckpt_verbose'],
        save_weights_only=config['saving']['ckpt_weights_only'],
    )

    early_stop_callback = EarlyStopping(
        monitor=config['saving']['ckpt_monitor'],
        min_delta=0.00,
        patience=config['saving']['ckpt_earlystopping_patience'],
        mode=config['saving']['ckpt_monitor_mode'],
    )

    prog_rate = TQDMProgressBar(refresh_rate=config['saving']["progress_rate"])

    callbacks = [
        ckpt_callback,
        early_stop_callback,
        prog_rate,
    ]

    logger = TensorBoardLogger(
        save_dir=out_dir,
        name=Path("tensorboard_logs_" + config['paths']["out_model_name"]).as_posix(),
    )

    loggers = [logger]

    trainer = Trainer(
        accelerator=config['hardware']["accelerator"],
        devices=config['hardware']["gpus_per_node"],
        strategy=config['hardware']["strategy"],
        num_nodes=config['hardware']["num_nodes"],
        max_epochs=config['hyperparams']["num_epochs"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=config['saving']["enable_progress_bar"],
    )

    if config['tasks']['train_tasks']['resume_training_from_ckpt']:
        print('---------------------------------------------------------------')
        print('------------- RESUMING TRAINING FROM CKPT_PATH ----------------')
        print('---------------------------------------------------------------')
        checkpoint = torch.load(config['paths']['ckpt_model_path'])
        trainer.fit(seg_module, datamodule=data_module, ckpt_path=config['paths']['ckpt_model_path'])
    else:
        trainer.fit(seg_module, datamodule=data_module)

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
        devices=config['hardware']["gpus_per_node"],
        strategy=config['hardware']["strategy"],
        num_nodes=config['hardware']["num_nodes"],
        callbacks=[writer_callback],
        enable_progress_bar=config['saving']["enable_progress_bar"],
        logger=False,
    )

    trainer.predict(seg_module, datamodule=data_module, return_predictions=False)
