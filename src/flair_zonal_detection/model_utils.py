import rasterio
from copy import deepcopy
from typing import Dict, Any

from flair_hub.models.flair_model import FLAIR_HUB_Model
from flair_hub.models.checkpoint import load_checkpoint


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
    patch_sizes = {}
    target_res = config['reference_resolution']

    for mod, active in config['modalities']['inputs'].items():
        if not active:
            continue
        mod_res = get_resolution(config['modalities'][mod]['input_img_path'])
        scale = mod_res / target_res
        patch_sizes[mod] = int(round(config['img_pixels_detection'] / scale))

    print('PATCH SIZES ---> ', patch_sizes)

    return patch_sizes


def prepare_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the exact model configuration saved by training."""
    cfg = deepcopy(config["runtime_model_config"])
    cfg["paths"]["ckpt_model_path"] = config["model_weights"]
    cfg["tasks"]["train"] = False
    cfg["models"]["monotemp_model"].setdefault("encoder", {})["weights"] = None
    cfg["modalities"]["aux_loss"] = {
        modality: False for modality in cfg["modalities"]["aux_loss"]
    }
    return cfg


def build_inference_model(config: Dict[str, Any], patch_sizes: Dict[str, int]) -> FLAIR_HUB_Model:
    """
    Build and load the FLAIR inference model from configuration and checkpoint.
    """
    model_cfg = prepare_model_config(config)
    model = FLAIR_HUB_Model(config=model_cfg, img_input_sizes=patch_sizes)
    load_checkpoint(model_cfg, model)
    return model.eval()
