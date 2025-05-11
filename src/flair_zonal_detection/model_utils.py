import rasterio
from copy import deepcopy
from src.flair_hub.models.flair_model import FLAIR_INC_Model
from src.flair_hub.models.checkpoint import load_checkpoint


def get_resolution(path):
    with rasterio.open(path) as src:
        return abs(src.res[0])  # assumes square pixels


def compute_patch_sizes(config):
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



def prepare_model_config(config):
    cfg = deepcopy(config)

    # Wrap monotemp_arch into training-style structure
    if 'models' not in cfg:
        cfg['models'] = {}

    if 'monotemp_arch' in config:
        cfg['models']['monotemp_model'] = {
            'arch': config['monotemp_arch'],
            'new_channels_init_mode': 'random'  # default
        }

    if 'multitemp_model_ref_date' in config:
        cfg['models']['multitemp_model'] = {
            'ref_date': config['multitemp_model_ref_date'],
            'encoder_widths': [64, 64, 64, 128], 'decoder_widths': [32, 32, 64, 128],
            'out_conv': [32, 19], 'str_conv_k': 3, 'str_conv_s': 1, 'str_conv_p': 1,
            'agg_mode': "att_group", 'encoder_norm': "group",
            'n_head': 16, 'd_model': 256, 'd_k': 4,
            'pad_value': 0, 'padding_mode': "reflect"
        }

    # Labels
    if "labels" not in cfg:
        cfg["labels"] = [t["name"] for t in cfg["tasks"] if t.get("active", False)]
    if "labels_configs" not in cfg:
        cfg["labels_configs"] = {
            task["name"]: {
                "value_name": list(task["class_names"].values())
            }
            for task in cfg["tasks"]
            if task.get("active", False)
        }
    # Input channels for all modalities (active or not)
    if "inputs_channels" not in cfg["modalities"]:
        cfg["modalities"]["inputs_channels"] = {
            mod: cfg["modalities"].get(mod, {}).get("channels", [])
            for mod in cfg["modalities"]["inputs"]
        }
    # Auxiliary loss (default to False)
    if "aux_loss" not in cfg["modalities"]:
        cfg["modalities"]["aux_loss"] = {
            mod: False for mod in cfg["modalities"]["inputs"]
        }
    # Ensure checkpoint path is placed where model expects it
    cfg.setdefault("paths", {})["ckpt_model_path"] = config["model_weights"]

    return cfg


def build_inference_model(config, patch_sizes):

    # Suppose you only use AERIAL_RGBI (a monotemp modality)
    model_cfg = prepare_model_config(config)
    model = FLAIR_INC_Model(config=model_cfg, img_input_sizes=patch_sizes)
    load_checkpoint(model_cfg, model)

    return model.eval()