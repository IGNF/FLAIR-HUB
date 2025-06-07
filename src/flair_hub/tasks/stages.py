import torch
import os
import datetime

from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict
from pytorch_lightning import seed_everything

from flair_hub.writer.prediction_writer import PredictionWriter
from flair_hub.tasks.trainers import train, predict
from flair_hub.models.checkpoint import load_checkpoint
from flair_hub.tasks.module_setup import get_input_img_sizes, build_segmentation_module



def training_stage(
    config: Dict,
    data_module,
    out_dir: Path
) -> OrderedDict:
    """
    Conducts the training stage of the model: sets up the training environment, loads the model weights 
    from a checkpoint if available, trains the model, and logs the training information.
    Args:
        config (Dict): Configuration dictionary containing parameters for the task.
        data_module: Data module providing training, validation, and testing data.
        out_dir (Path): Path object representing the output directory for saving checkpoints and logs.
    Returns:
        OrderedDict: The state dictionary of the best-performing trained model.
    """
    start = datetime.datetime.now()

    seed_everything(config['hyperparams']['seed'], workers=True)

    in_img_sizes = get_input_img_sizes(config, data_module, stage='fit')

    seg_module = build_segmentation_module(config, in_img_sizes, stage='train')

    if config['tasks']['train_tasks']['init_weights_only_from_ckpt']:
        load_checkpoint(config, seg_module, exit_on_fail=True)

    ckpt_callback = train(config, data_module, seg_module, out_dir)

    best_trained_state_dict = torch.load(ckpt_callback.best_model_path, map_location=torch.device('cpu'))['state_dict']

    end = datetime.datetime.now()
    training_time_seconds = end - start
    training_time_seconds = training_time_seconds.total_seconds()

    print(f"\n[Training finished in {str(timedelta(seconds=training_time_seconds))} HH:MM:SS with "
          f"{config['hardware']['num_nodes']} nodes and {config['hardware']['gpus_per_node']} gpus per node]")
    print(f"Model path: {os.path.join(out_dir, 'checkpoints')}\n\n")
    print('-' * 40)

    return best_trained_state_dict



def predict_stage(
    config: Dict,
    data_module,
    out_dir_predict: Path,
    trained_state_dict: Optional[OrderedDict] = None
) -> None:
    """
    Conducts the prediction stage of the model: runs inference and optionally computes metrics.
    Handles prediction, evaluation from disk, or notifies if neither task is enabled.
    Args:
        config (Dict): Configuration dictionary containing prediction and evaluation settings.
        data_module: Data module providing data for prediction.
        out_dir_predict (Path): Directory where predictions and results are saved.
        trained_state_dict (Optional[OrderedDict]): Optional state dictionary of the trained model.
            If not provided, model weights will be loaded from checkpoint.
    Returns:
        None
    """

    out_dir_predict = Path(out_dir_predict)

    if config['tasks'].get("metrics_only", False) and not config['tasks'].get("predict", False):
        print("[ ] Running in metrics-only mode: loading predictions from disk . . .")
        writer = PredictionWriter(config, out_dir_predict, write_interval="batch")
        writer.load_predictions_and_compute_metrics()
        return 

    if config['tasks'].get("predict", False):
        in_img_sizes = get_input_img_sizes(config, data_module, stage="predict")
        seg_module = build_segmentation_module(config, in_img_sizes, stage='predict')

        if config['tasks']['train']:
            seg_module.load_state_dict(trained_state_dict, strict=False)
        else:
            load_checkpoint(config, seg_module)

        print("[ ] Running inference and metrics calculation . . .")
        predict(config, data_module, seg_module, out_dir_predict)

    if not config['tasks'].get("predict") and not config['tasks'].get("metrics_only"):
        print("[ ] Neither 'predict' nor 'metrics_only' is enabled. Finishing.")
