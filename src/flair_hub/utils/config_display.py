from typing import Optional
from pytorch_lightning.utilities.rank_zero import rank_zero_only  


@rank_zero_only
def print_recap(config: dict,
                dict_train: Optional[dict] = None,
                dict_val: Optional[dict] = None,
                dict_test: Optional[dict] = None) -> None:
    """
    Prints content of the given config using a tree structure.
    Args:
        config (dict): The configuration dictionary.
        dict_train (Optional[dict]): Training data dictionary.
        dict_val (Optional[dict]): Validation data dictionary.
        dict_test (Optional[dict]): Test data dictionary.
    Filters some sections if verbose_config is False.
    """

    def walk_config(d: dict, prefix: str = '', filter_section: bool = False,
                    active_inputs: Optional[set] = None, parent_key: Optional[str] = None) -> None:
        """
        Recursive config printer with optional filtering logic for inactive inputs.
        """
        for k, v in d.items():

            if active_inputs is not None:
                if parent_key in {"inputs_channels", "aux_loss", "modality_dropout"}:
                    if k not in active_inputs:
                        continue
                elif parent_key == "normalization":
                    if k.endswith("_means") or k.endswith("_stds"):
                        base = k.replace("_means", "").replace("_stds", "")
                        if base not in active_inputs:
                            continue

            if isinstance(v, dict):
                if filter_section and all(x in [False, 0, None, "", [], {}] for x in v.values()):
                    continue
                print(f'{prefix}|- {k}:')
                walk_config(v, prefix + '|   ', filter_section, active_inputs, parent_key=k)
            elif isinstance(v, list):
                if not filter_section or v:
                    print(f'{prefix}|- {k}: {v}')
            else:
                if not filter_section or v not in [False, 0, None, "", [], {}]:
                    print(f'{prefix}|- {k}: {v}')

    verbose = config.get("saving", {}).get("verbose_config", True)
    inputs = config.get("modalities", {}).get("inputs", {})
    active_inputs = {k for k, v in inputs.items() if v}

    print("Configuration Tree:")
    for section_key, section_value in config.items():
        if isinstance(section_value, dict):
            print(f'|- {section_key}:')
            if section_key == "modalities":
                walk_config(section_value, prefix='|   ',
                            filter_section=not verbose,
                            active_inputs=active_inputs)
            else:
                walk_config(section_value, prefix='|   ',
                            filter_section=not verbose)
        else:
            print(f'|- {section_key}: {section_value}')

    list_keys = [
        'AERIAL_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV', 'SPOT_RGBI',
        'SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS'
    ]
    for label in config.get('labels', []):
        list_keys.append(label)

    print('\n[---DATA SPLIT---]')
    if config['tasks'].get('train', False):
        print('[TRAIN]')
        for key in list_keys:
            if dict_train and dict_train.get(key):
                print(f"- {key:20s}: {len(dict_train[key])} samples")
        print('[VAL]')
        for key in list_keys:
            if dict_val and dict_val.get(key):
                print(f"- {key:20s}: {len(dict_val[key])} samples")

    if config['tasks'].get('predict', False):
        print('[TEST]')
        for key in list_keys:
            if dict_test and dict_test.get(key):
                print(f"- {key:20s}: {len(dict_test[key])} samples")
