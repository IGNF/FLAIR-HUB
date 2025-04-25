import yaml
import os
import shutil

from pathlib import Path
from typing import Dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only  



def read_configs(folder_path: str) -> Dict[str, dict]:
    """
    Reads and combines all YAML configuration files in the given folder.

    Args:
        folder_path (str): The folder containing the YAML configuration files.

    Returns:
        Dict[str, dict]: A combined dictionary with the contents of all YAML files.
    """
    combined_config = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
                combined_config.update(config)
    return combined_config


def setup_environment(args) -> tuple:
    """
    This function reads the configuration file, creates the output directory, 
    and sets up the logger.

    Args:
        args: Command-line arguments.

    Returns:
        tuple: Contains the config dictionary and the output directory path.
    """
    config = read_configs(args.conf_folder)
    out_dir = Path(config['paths']["out_folder"], config['paths']["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return config, out_dir


@rank_zero_only
def copy_csv_and_config(config: dict, out_dir: Path, args) -> None:
    """
    Copy the CSV files and configuration file to the output directory.

    Args:
        config (dict): Configuration dictionary.
        out_dir (Path): Output directory path.
        args: Command-line arguments.
    """
    csv_copy_dir = Path(out_dir, 'used_csv_and_config')
    csv_copy_dir.mkdir(parents=True, exist_ok=True)

    if config["tasks"]["train"]:
        shutil.copy(config["paths"]["train_csv"], csv_copy_dir)
        shutil.copy(config["paths"]["val_csv"], csv_copy_dir)
    
    if config["tasks"]["predict"]:
        shutil.copy(config["paths"]["test_csv"], csv_copy_dir)

    shutil.copytree(args.conf_folder, csv_copy_dir, dirs_exist_ok=True)
