import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only  


def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
     

@rank_zero_only
def print_recap(config: dict, dict_train=None, dict_val=None, dict_test=None) -> None:
    """Log content of given config using a tree structure."""

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
    # Log data split information
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




