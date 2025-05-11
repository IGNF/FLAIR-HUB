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
        'margin', 'modalities', 'tasks', 'reference_modality_for_slicing'
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    ref_mod = config['reference_modality_for_slicing']
    if ref_mod not in config['modalities'] or 'input_img_path' not in config['modalities'][ref_mod]:
        raise ValueError("Reference modality for slicing is not properly defined.")

    if not os.path.isfile(config['model_weights']):
        raise FileNotFoundError(f"Model weights not found at: {config['model_weights']}")

    os.makedirs(config['output_path'], exist_ok=True)



def summarize_config(config: dict) -> None:
    """
    Print a summary of the configuration settings.
    """
    ref_mod = config['reference_modality_for_slicing']
    used_mods = ', '.join([m for m, active in config['modalities']['inputs'].items() if active])
    active_tasks = ', '.join([t['name'] for t in config['tasks'] if t['active']])
    device = "cuda" if config.get('use_gpu', False) else "cpu"

    print(f"""
##############################################
FLAIR-HUB ZONE DETECTION
##############################################
|→ Output path        : {config['output_path']}
|→ Output file name   : {config['output_name']}.tif

|→ Reference modality : {ref_mod}
|→ Input image path   : {config['modalities'][ref_mod]['input_img_path']}
|→ Tile size (px)     : {config['img_pixels_detection']}
|→ Margin (px)        : {config['margin']}

|→ Modalities used    : {used_mods}
|→ Tasks active       : {active_tasks}
|→ Output type        : {config['output_type']}

|→ Checkpoint path    : {config['model_weights']}
|→ Device             : {device}
|→ Batch size         : {config['batch_size']}
|→ Num workers        : {config['num_worker']}
""")