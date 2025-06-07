import argparse
import sys

from pathlib import Path

from flair_hub.tasks.stages import training_stage, predict_stage
from flair_hub.tasks.module_setup import build_data_module
from flair_hub.data.utils_data.paths import get_datasets
from flair_hub.utils.messaging import start_msg, end_msg, Logger
from flair_hub.utils.config_io import setup_environment, copy_csv_and_config
from flair_hub.utils.config_display import print_recap


argParser = argparse.ArgumentParser()
argParser.add_argument("--config", help="Path to the .yaml config file", required=True)


def main():
    """
    Main function to set up the training and prediction process. It reads the config file, sets up the output folder, 
    initiates the training and prediction stages, and tracks emissions if enabled.
    """

    args = argParser.parse_args()
    config, out_dir = setup_environment(args)
    sys.stdout = Logger(
        Path(config['paths']["out_folder"], config['paths']["out_model_name"], f'flair-compute{config["paths"]["out_model_name"]}.log').as_posix())

    start_msg()

    # Define datasets
    dict_train, dict_val, dict_test = get_datasets(config)
    print_recap(config, dict_train, dict_val, dict_test)

    # Copy relevant files for tracking
    if config['saving']["cp_csv_and_conf_to_output"]:
        copy_csv_and_config(config, out_dir, args)

    # Get LightningDataModule
    dm = build_data_module(config, dict_train=dict_train, dict_val=dict_val, dict_test=dict_test)

    # Initialize variable for weights
    trained_state_dict = None

    # Training
    if config['tasks']['train']:
        trained_state_dict = training_stage(config, dm, out_dir)

    # Inference
    if config['tasks'].get('predict') or config['tasks'].get('metrics_only'):
        out_dir_predict = Path(out_dir, 'results_' + config['paths']["out_model_name"])
        out_dir_predict.mkdir(parents=True, exist_ok=True)
        predict_stage(config, dm, out_dir_predict, trained_state_dict)
    else:
        print("[WARNING] Neither prediction nor metrics_only was enabled. Finishing.")

    end_msg()

if __name__ == "__main__":
    main()
