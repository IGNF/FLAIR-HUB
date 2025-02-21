import datetime
import os
import argparse
import torch
import sys
from pathlib import Path
from datetime import timedelta
from pytorch_lightning import seed_everything

from flair_inc.tasks import train, predict
from flair_inc.utils.utils_tasks import get_data_module, get_segmentation_module, get_input_img_sizes
from flair_inc.utils.utils_data import load_checkpoint, get_datasets
from flair_inc.utils.utils import setup_environment, Logger, copy_csv_and_config, print_recap, emission_tracking_summary

# from codecarbon import OfflineEmissionsTracker

argParser = argparse.ArgumentParser()
argParser.add_argument("--conf_folder", help="Path to the .yaml config file", required=True)


def training_stage(config, data_module, out_dir):
    """
    Conducts the training stage of the model: sets up the training environment, loads the model weights 
    from a checkpoint if available, trains the model, and logs the training information.

    Args:
        config (dict): Configuration dictionary containing parameters for the task.
        data_module: Data module for training, validation, and testing.
        out_dir (Path): Path object representing the output directory.

    Returns:
        OrderedDict: The state dictionary of the trained model.
    """
    start = datetime.datetime.now()

    seed_everything(config['hyperparams']['seed'], workers=True)

    in_img_sizes = get_input_img_sizes(config, data_module)

    seg_module = get_segmentation_module(config, in_img_sizes, stage='train')

    if config['tasks']['train_tasks']['init_weights_only_from_ckpt']:
        load_checkpoint(config, seg_module, exit_on_fail=False)

    ckpt_callback = train(config, data_module, seg_module, out_dir)

    best_trained_state_dict = torch.load(ckpt_callback.best_model_path, map_location=torch.device('cpu'))['state_dict']

    end = datetime.datetime.now()
    inference_time_seconds = end - start
    inference_time_seconds = inference_time_seconds.total_seconds()

    print(f"\n[Training finished in {str(timedelta(seconds=inference_time_seconds))} HH:MM:SS with "
          f"{config['hardware']['num_nodes']} nodes and {config['hardware']['gpus_per_node']} gpus per node]")
    print(f"Model path: {os.path.join(out_dir, 'checkpoints')}\n\n")
    print('-' * 40)

    return best_trained_state_dict


def predict_stage(config, data_module, out_dir_predict, trained_state_dict=None):
    """
    Conducts the prediction stage of the model: sets up the prediction environment, loads the model weights 
    from the training stage or a checkpoint file, and makes predictions.

    Args:
        config (dict): Configuration dictionary containing parameters for the task.
        data_module: Data module for training, validation, and testing.
        out_dir_predict (Path): Path object representing the output directory for predictions.
        trained_state_dict (OrderedDict, optional): The state dictionary of the trained model. Defaults to None.
    """
    in_img_sizes = get_input_img_sizes(config, data_module)

    seg_module = get_segmentation_module(config, in_img_sizes, stage='predict')
    if config['tasks']['train']:
        seg_module.load_state_dict(trained_state_dict, strict=False)
    else:
        load_checkpoint(config, seg_module)
    predict(config, data_module, seg_module, out_dir_predict)


def main():
    """
    Main function to set up the training and prediction process. It reads the config file, sets up the output folder, 
    initiates the training and prediction stages, and tracks emissions if enabled.
    """
    print('######## LAUNCHING ########')
    print('###########################')
    print('###########################')

    # Read config and create output folder
    args = argParser.parse_args()
    config, out_dir = setup_environment(args)

#    if config['saving']['codecarbon']:  # TODO CODECARBON
#        node_id = os.getenv("SLURM_NODEID", "0")
#        output_emission_dir = Path(out_dir) / f"node_{node_id}_emissions"
#        output_emission_dir.mkdir(parents=True, exist_ok=True)
#        tracker = OfflineEmissionsTracker(
#            output_dir=str(output_emission_dir),
#            project_name=f'flair-inc | node_{node_id}',
#            log_level="error",
#            country_iso_code='FRA'
#        )
#        tracker.start()

    # Custom Logger for console/logfile output
    sys.stdout = Logger(
        Path(config['paths']["out_folder"], config['paths']["out_model_name"], 'flair-compute.log').as_posix())
    print(datetime.datetime.now().strftime("Starting: %Y-%m-%d  %H:%M") + '\n')

    # Define datasets
    dict_train, dict_val, dict_test = get_datasets(config)
    print(dict_val.keys())
    print_recap(config, dict_train, dict_val, dict_test)

    # Copy relevant files for tracking
    if config['saving']["cp_csv_and_conf_to_output"]:
        copy_csv_and_config(config, out_dir, args)

    # Get LightningDataModule
    dm = get_data_module(config, dict_train=dict_train, dict_val=dict_val, dict_test=dict_test)

    # Initialize variable for weights
    trained_state_dict = None

    # Training
    if config['tasks']['train']:
        trained_state_dict = training_stage(config, dm, out_dir)

    # Inference
    if config['tasks']['predict']:
        out_dir_predict = Path(out_dir, 'results_' + config['paths']["out_model_name"])
        out_dir_predict.mkdir(parents=True, exist_ok=True)
        predict_stage(config, dm, out_dir_predict, trained_state_dict)

#    if config['saving']['codecarbon']:  # TODO CODECARBON
#        tracker.stop()
#        emission_tracking_summary(out_dir)


if __name__ == "__main__":
    main()
