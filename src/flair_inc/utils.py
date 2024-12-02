import yaml
import sys
import shutil
from pathlib import Path
from pytorch_lightning.utilities.rank_zero import rank_zero_only  


def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
     

def setup_environment(args):
    """
    This function reads the configuration file, creates the output directory, 
    and sets up the logger.
    """
    config = read_config(args.conf)
    out_dir = Path(config['paths']["out_folder"], config['paths']["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return config, out_dir 


@rank_zero_only
class Logger(object):
    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8') 
        self.encoding = self.terminal.encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


@rank_zero_only
def copy_csv_and_config(config, out_dir, args):
    """
    Copy the CSV files and configuration file to the output directory.
    """
    csv_copy_dir = Path(out_dir, 'used_csv_and_config')
    csv_copy_dir.mkdir(parents=True, exist_ok=True)
    if config["tasks"]["train"]:
        shutil.copy(config["paths"]["train_csv"], csv_copy_dir)
        shutil.copy(config["paths"]["val_csv"], csv_copy_dir)
    if config["tasks"]["predict"]: shutil.copy(config["paths"]["test_csv"], csv_copy_dir)
    shutil.copy(args.conf, csv_copy_dir)


@rank_zero_only
def print_recap(config: dict, dict_train=None, dict_val=None, dict_test=None) -> None:
    """Print content of given config using a tree structure."""

    def walk_config(config: dict, prefix=''):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, dict):
                print(f'{prefix}|- {group_name}:')
                walk_config(group_option, prefix=prefix+'|   ')
            elif isinstance(group_option, list):
                print(f'{prefix}|- {group_name}: {group_option}')
            else:
                print(f'{prefix}|- {group_name}: {group_option}')

    print('Configuration Tree:')
    walk_config(config, '')

    list_keys = ['AERIAL_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV', 'SPOT_RGBI', 'SENTINEL2_TS', 'SENTINEL1-ASC_TS',
                'SENTINEL1-DESC_TS', 'LABELS']
    print('[---DATA SPLIT---]')
    if config['tasks']['train']:
        print('[TRAIN]')
        for key in list_keys:
            if dict_train[key] != []:  
                print(f"- {key:20s}: {'':3s}{len(dict_train[key])} samples")        
        print('[VAL]')
        for key in list_keys:
            if dict_val[key] != []:  
                print(f"- {key:20s}: {'':3s}{len(dict_val[key])} samples")
    if config['tasks']['predict']:
        print('[TEST]')
        for key in list_keys:
            if dict_test[key] != []:  
                print(f"- {key:20s}: {'':3s}{len(dict_test[key])} samples")




