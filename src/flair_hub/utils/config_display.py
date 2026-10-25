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

    def walk_config(d: dict, prefix: str = '') -> None:
        """Print a resolved configuration mapping recursively."""
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix}|- {k}:')
                walk_config(v, prefix + '|   ')
            elif isinstance(v, list):
                print(f'{prefix}|- {k}: {v}')
            else:
                print(f'{prefix}|- {k}: {v}')

    verbose = config.get("saving", {}).get("verbose_config", True)
    display_config = config.get("_resolved_config", config)
    inputs = config.get("modalities", {}).get("inputs", {})
    active_inputs = {k for k, v in inputs.items() if v}

    print("Configuration summary (full details: resolved_config.yaml):")
    if verbose:
        for section_key, section_value in display_config.items():
            if isinstance(section_value, dict):
                print(f'|- {section_key}:')
                walk_config(section_value, prefix='|   ')
            else:
                print(f'|- {section_key}: {section_value}')
    else:
        public_model = display_config["model"]["monotemporal"]
        print(f"- experiment              : {display_config['experiment']['name']}")
        print(
            f"- encoder / decoder       : {public_model['encoder']['name']} / "
            f"{public_model['decoder']['name']} ({public_model['decoder']['provider']})"
        )
        for name in sorted(active_inputs):
            modality = display_config["data"]["modalities"][name]
            print(
                f"- active modality         : {name}, bands "
                f"{modality.get('input_band_order', 'derived')}, "
                f"normalization {modality['normalization']['type']}"
            )
        print(f"- active outputs          : {', '.join(config.get('labels', []))}")
        print(
            f"- augmentations           : "
            f"{display_config['data']['augmentation']['enabled']}"
        )
        if display_config["data"]["augmentation"]["enabled"]:
            radiometry = display_config["data"]["augmentation"].get(
                "radiometry", {}
            )
            for name, radiometry_cfg in radiometry.items():
                if not radiometry_cfg.get("enabled", False):
                    continue
                band_order = display_config["data"]["modalities"][name].get(
                    "input_band_order", []
                )
                print(
                    f"- radiometric augmentation: {name}, "
                    f"{radiometry_cfg.get('mode', 'hsv')}, bands {band_order}"
                )

    hyper = config.get("hyperparams", {})
    hardware = config.get("hardware", {})
    world_size = int(hardware.get("num_nodes", 1)) * int(
        hardware.get("gpus_per_node", 1)
    )
    effective_batch = int(hyper.get("batch_size", 1)) * world_size
    duration_unit = hyper.get("duration_unit", "epochs")
    duration_total = (
        hyper.get("max_steps") if duration_unit == "steps"
        else hyper.get("num_epochs")
    )
    print("\n[---RESOLVED TRAINING PLAN---]")
    print(f"- duration               : {duration_total:,} {duration_unit}")
    print(
        f"- validation interval    : every "
        f"{int(hyper.get('validation_every', 1)):,} {duration_unit}"
    )
    print(
        f"- resources              : {hardware.get('num_nodes', 1)} node(s) × "
        f"{hardware.get('gpus_per_node', 1)} GPU"
    )
    print(
        f"- effective global batch : {hyper.get('batch_size', 1)} per GPU × "
        f"{world_size} GPU = {effective_batch}"
    )
    if duration_unit == "steps" and dict_train:
        sample_count = next(
            (
                len(dict_train[label])
                for label in config.get("labels", [])
                if dict_train.get(label)
            ),
            0,
        )
        if sample_count:
            equivalent_passes = duration_total * effective_batch / sample_count
            print(
                f"- equivalent data passes : {equivalent_passes:.2f} "
                f"over {sample_count:,} patches (informational)"
            )

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
                print(f"- {key:20s}: {len(dict_train[key]):,} samples")
        print('[VAL]')
        for key in list_keys:
            if dict_val and dict_val.get(key):
                print(f"- {key:20s}: {len(dict_val[key]):,} samples")

    if config['tasks'].get('predict', False):
        print('[TEST]')
        for key in list_keys:
            if dict_test and dict_test.get(key):
                print(f"- {key:20s}: {len(dict_test[key]):,} samples")

    print()
