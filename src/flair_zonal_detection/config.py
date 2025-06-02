import os, sys
import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only



def load_config(path: str) -> dict:
    """
    Load YAML configuration from the given file path.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    """
    Validate that required configuration keys are present and valid.
    """
    required_keys = [
        'output_path', 'output_name', 'model_weights', 'img_pixels_detection',
        'margin', 'modalities', 'tasks', 'output_px_meters'
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if not os.path.isfile(config['model_weights']):
        raise FileNotFoundError(f"Model weights not found at: {config['model_weights']}")

    os.makedirs(config['output_path'], exist_ok=True)



def config_recap_1(config: dict) -> None: 
    
    used_mods = ', '.join([m for m, active in config['modalities']['inputs'].items() if active])
    active_tasks = ', '.join([t['name'] for t in config['tasks'] if t['active']])
    device = "cuda" if config.get('use_gpu', False) else "cpu"

    print(f"""
##############################################
FLAIR-HUB ZONE DETECTION
##############################################
|→ Output path            : {config['output_path']}
|→ Output file name       : {config['output_name']}.tif

|→ Modalities used        : {used_mods}
|→ Tasks active           : {active_tasks}
|→ Output type            : {config['output_type']}

|→ Checkpoint path        : {config['model_weights']}
|→ Device                 : {device}
|→ Batch size             : {config['batch_size']}
|→ Num workers            : {config['num_worker']}
""")


def config_recap_2(config: dict) -> None:
    """
    Print resolution and tiling details with aligned formatting.
    """
    def fmt(val, precision=5):
        return f"{val:.{precision}f}".rstrip('0').rstrip('.') if isinstance(val, float) else str(val)
    
    shape = config.get('image_shape_px', {})
    res = config['reference_resolution']
    if shape:
        height_px = shape['height']
        width_px = shape['width']
        height_m = height_px * res
        width_m = width_px * res
        print(f"|→ Image size (px)         : {height_px} (H) × {width_px} (W)")
        print(f"|→ Image size (meters)     : {fmt(height_m, 2)} m (H) × {fmt(width_m, 2)} m (W)")

    print(f"|→ Reference resolution    : {fmt(res)} m/px")
    print(f"|→ Output resolution       : {fmt(config['output_px_meters'])} m/px\n")

    patch_px = config['img_pixels_detection']
    patch_m = patch_px * res
    margin_px = config['margin']
    margin_m = margin_px * res

    print(f"|→ Patch size              : {patch_px} px → {fmt(patch_m, 2)} m")
    print(f"|→ Margin size             : {margin_px} px → {fmt(margin_m, 2)} m")
    print("|→ Modalities resolution   :")
    for mod, r in config['modality_resolutions'].items():
        print(f"   - {mod:15}: {fmt(r)} m/px")

    print("=" * 46 + "\n")


