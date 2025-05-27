import rasterio
from copy import deepcopy
from typing import Dict, Any

from src.flair_hub.models.flair_model import FLAIR_INC_Model
from src.flair_hub.models.checkpoint import load_checkpoint


def get_resolution(path: str) -> float:
    """
    Return the pixel resolution from a raster file (assumes square pixels).
    """
    with rasterio.open(path) as src:
        return abs(src.res[0])


def compute_patch_sizes(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Compute patch size for each modality based on the reference modality's resolution.
    """
    ref_mod = config['reference_modality_for_slicing']
    ref_path = config['modalities'][ref_mod]['input_img_path']
    ref_patch = config['img_pixels_detection']
    ref_res = get_resolution(ref_path)

    patch_sizes = {}
    for mod, active in config['modalities']['inputs'].items():
        if not active:
            continue
        mod_path = config['modalities'][mod]['input_img_path']
        mod_res = get_resolution(mod_path)
        scale = mod_res / ref_res
        patch_sizes[mod] = int(round(ref_patch / scale))

    return patch_sizes


def prepare_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a full model configuration dictionary from the base config.
    Adds model, label, input, and checkpoint parameters.
    """
    cfg = deepcopy(config)

    # Wrap architecture if needed
    if 'models' not in cfg:
        cfg['models'] = {}

    if 'monotemp_arch' in config:
        cfg['models']['monotemp_model'] = {
            'arch': config['monotemp_arch'],
            'new_channels_init_mode': 'random'
        }

    if 'multitemp_model_ref_date' in config:
        cfg['models']['multitemp_model'] = {
            'ref_date': config['multitemp_model_ref_date'],
            'encoder_widths': [64, 64, 64, 128],
            'decoder_widths': [32, 32, 64, 128],
            'out_conv': [32, 19],
            'str_conv_k': 3,
            'str_conv_s': 1,
            'str_conv_p': 1,
            'agg_mode': "att_group",
            'encoder_norm': "group",
            'n_head': 16,
            'd_model': 256,
            'd_k': 4,
            'pad_value': 0,
            'padding_mode': "reflect"
        }

    # Labels setup
    cfg.setdefault("labels", [t["name"] for t in cfg["tasks"] if t.get("active", False)])
    cfg.setdefault("labels_configs", {
        task["name"]: {"value_name": list(task["class_names"].values())}
        for task in cfg["tasks"] if task.get("active", False)
    })

    # Input channels for all declared modalities
    cfg["modalities"].setdefault("inputs_channels", {
        mod: cfg["modalities"].get(mod, {}).get("channels", [])
        for mod in cfg["modalities"]["inputs"]
    })

    # Auxiliary loss default values
    cfg["modalities"].setdefault("aux_loss", {
        mod: False for mod in cfg["modalities"]["inputs"]
    })

    # === Patch: Generate pre_processings block ===
    dem_cfg = cfg["modalities"].get("DEM_ELEV", {})
    pre_proc_defaults = {
        "calc_elevation": dem_cfg.get("calc_elevation", False),
        "calc_elevation_stack_dsm": dem_cfg.get("calc_elevation_stack_dsm", False),
        "filter_sentinel2": False,
        "filter_sentinel2_max_cloud": 100,
        "filter_sentinel2_max_snow": 100,
        "filter_sentinel2_max_frac_cover": 1.0,
        "temporal_average_sentinel2": False,
        "temporal_average_sentinel1": False,
        "use_augmentation": False
    }
    cfg["modalities"].setdefault("pre_processings", pre_proc_defaults)

    # Ensure checkpoint path is passed
    cfg.setdefault("paths", {})["ckpt_model_path"] = config["model_weights"]

    return cfg


def build_inference_model(config: Dict[str, Any], patch_sizes: Dict[str, int]) -> FLAIR_INC_Model:
    """
    Build and load the FLAIR inference model from configuration and checkpoint.
    """
    model_cfg = prepare_model_config(config)
    model = FLAIR_INC_Model(config=model_cfg, img_input_sizes=patch_sizes)
    load_checkpoint(model_cfg, model)
    return model.eval()
