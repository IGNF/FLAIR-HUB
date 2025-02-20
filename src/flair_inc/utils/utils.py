import yaml
import os, sys
import shutil
import pandas as pd

from pathlib import Path
from typing import Dict, Optional, Union
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
class Logger:
    def __init__(self, filename: str = 'Default.log') -> None:
        """
        Initializes a custom logger to output to both terminal and log file.

        Args:
            filename (str): Name of the log file.
        """
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.encoding = self.terminal.encoding

    def write(self, message: str) -> None:
        """
        Writes the log message to both the terminal and the log file.

        Args:
            message (str): The message to be logged.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """
        Flushes the log file to ensure all data is written.
        """
        self.log.flush()


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


@rank_zero_only
def print_recap(config: dict, dict_train: Optional[dict] = None,
                dict_val: Optional[dict] = None, dict_test: Optional[dict] = None) -> None:
    """
    Prints content of the given config using a tree structure.

    Args:
        config (dict): The configuration dictionary.
        dict_train (Optional[dict]): Training data dictionary.
        dict_val (Optional[dict]): Validation data dictionary.
        dict_test (Optional[dict]): Test data dictionary.
    """

    def walk_config(config: dict, prefix: str = '') -> None:
        """
        Recursive function to accumulate and print configuration tree.
        Args:
            config (dict): The configuration dictionary.
            prefix (str): The current prefix to format the output.
        """
        for group_name, group_option in config.items():
            if isinstance(group_option, dict):
                print(f'{prefix}|- {group_name}:')
                walk_config(group_option, prefix=prefix + '|   ')
            elif isinstance(group_option, list):
                print(f'{prefix}|- {group_name}: {group_option}')
            else:
                print(f'{prefix}|- {group_name}: {group_option}')

    print('Configuration Tree:')
    walk_config(config, '')

    list_keys = [
        'AERIAL_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV', 'SPOT_RGBI', 'SENTINEL2_TS', 
        'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS'
    ]
    
    for k in config['labels']:
        list_keys.append(k)
    
    print('[---DATA SPLIT---]')
    if config['tasks']['train']:
        print('[TRAIN]')
        for key in list_keys:
            if dict_train.get(key, []):
                print(f"- {key:20s}: {'':3s}{len(dict_train[key])} samples")
        
        print('[VAL]')
        for key in list_keys:
            if dict_val.get(key, []):
                print(f"- {key:20s}: {'':3s}{len(dict_val[key])} samples")
    
    if config['tasks']['predict']:
        print('[TEST]')
        for key in list_keys:
            if dict_test.get(key, []):
                print(f"- {key:20s}: {'':3s}{len(dict_test[key])} samples")


@rank_zero_only
def emission_tracking_summary(out_dir: Union[str, Path]) -> None:
    """
    Summarizes the emissions data by aggregating logs from all nodes.
    This function reads emissions data from CSV files generated by 
    CodeCarbon's trackers for individual nodes. It aggregates the data 
    and prints a summary of total CO2 emissions, energy consumption, 
    and energy usage breakdown by CPU, GPU, and RAM.

    Args:
        out_dir (Union[str, Path]): Path to the output directory where emissions logs are stored.
    Returns:
        None
    """
    emission_logs_dir = Path(out_dir)
    emission_files = emission_logs_dir.glob("node_*/emissions.csv")

    # Load and combine all logs into a single DataFrame
    dfs = [pd.read_csv(file) for file in emission_files]
    aggregated = pd.concat(dfs).sum(numeric_only=True)

    # Print the aggregated emissions summary
    print("\n----- Total Emissions Summary -----")
    print(f"Total CO2 emissions: {aggregated['emissions']:.6f} kg CO2e")
    print(f"Total energy consumption: {aggregated['energy_consumed']:.6f} kWh")
    print(f"Total energy CPU: {aggregated['cpu_energy']:.6f} kWh")
    print(f"Total energy GPU: {aggregated['gpu_energy']:.6f} kWh")
    print(f"Total energy RAM: {aggregated['ram_energy']:.6f} kWh")
    print(f"Measures done: {aggregated['energy_consumed']:.6f}")
