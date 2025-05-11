import os, sys
import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@rank_zero_only
class Logger(object):
    def __init__(self, filename='inference.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.encoding = self.terminal.encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(config):
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



def summarize_config(config):
    ref_mod = config['reference_modality_for_slicing']

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

|→ Modalities used    : {', '.join([m for m, act in config['modalities']['inputs'].items() if act])}
|→ Tasks active       : {', '.join([t['name'] for t in config['tasks'] if t['active']])}
|→ Output type        : {config['output_type']}

|→ Checkpoint path    : {config['model_weights']}
|→ Device             : {"cuda" if config['use_gpu'] else "cpu"}
|→ Batch size         : {config['batch_size']}
|→ Num workers        : {config['num_worker']}

    """)
