import datetime
import os
import argparse 
import torch
import sys
import numpy as np

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import timedelta

from pytorch_lightning import seed_everything 
from pytorch_lightning.utilities.rank_zero import rank_zero_only  

from flair_inc.tasks import train, predict
from flair_inc.utils.utils_tasks import get_data_module, get_segmentation_module
from flair_inc.utils.utils_data import get_paths, get_sentinel_dates_mtd
from flair_inc.utils.utils import setup_environment, Logger, copy_csv_and_config, print_recap

from codecarbon import OfflineEmissionsTracker

argParser = argparse.ArgumentParser()
argParser.add_argument("--conf", help="Path to the .yaml config file", required=True)

   

def get_datasets(config: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Get the datasets for training, validation, and testing.
    Args:
        config (Dict[str, Any]): Configuration dictionary.
    Returns:
        Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]: Datasets for training, validation, and testing.
    """
    dict_train, dict_val, dict_test = None, None, None
    
    if config['tasks']['train']:
        dict_train = get_paths(config, split='train')
        dict_val   = get_paths(config, split='val')
        
    if config['tasks']['predict']: 
        dict_test  = get_paths(config, split='test')
    
    dates_s2, dates_s1asc, dates_s1desc = get_sentinel_dates_mtd(config)

    for d in [dict_train, dict_val, dict_test]: 
        if d is not None:
            d['DATES_S2'] = dates_s2
            d['DATES_S1_ASC'] = dates_s1asc
            d['DATES_S1_DESC'] = dates_s1desc
    
    return dict_train, dict_val, dict_test


@rank_zero_only
def load_checkpoint(conf, seg_module, exit_on_fail=False):
    """
    Load model weights from a checkpoint file and adjust final classification layers for new number of classes if needed.
    
    Parameters:
    config (dict): experiment config.
    seg_module: Segmentation module for training or prediction.
    exit_on_fail (bool): Whether to raise a SystemExit if the checkpoint file is invalid.
    """
    print()
    print('###############################################################')

    ckpt_file_path = conf['paths']['ckpt_model_path']
    num_classes = len(conf["classes"])
    
    # Ensure the checkpoint file path is valid
    if ckpt_file_path and os.path.isfile(ckpt_file_path):
        checkpoint = torch.load(ckpt_file_path, map_location="cpu")
        
        if ckpt_file_path.endswith('.ckpt'):
            state_dict = checkpoint.get("state_dict", checkpoint)
        elif ckpt_file_path.endswith('.pth') or ckpt_file_path.endswith('.pt'):
            state_dict = checkpoint
        else:
            print("Invalid file extension.")
            if exit_on_fail:
                raise SystemExit()
            return
        
        # Determine number of classes from checkpoint
        ckpt_num_classes = None
        for k, v in state_dict.items():
            if 'classifier.weight' in k or 'criterion.weight' in k:
                ckpt_num_classes = v.shape[0]
                break

        model_state_dict = seg_module.state_dict()
        
        # Load model weights if class numbers match
        if ckpt_num_classes is not None and ckpt_num_classes == num_classes:
            seg_module.load_state_dict(state_dict, strict=False)
            print('--------------- Loaded model weights from checkpoint with matching number of classes. ---------------')
        else:
            print(f'Number of classes in checkpoint ({ckpt_num_classes}) does not match the current number of classes ({num_classes}). Proceeding with modifications.')
            
            # Identify and exclude layers with mismatched shapes
            ignored_layers = [k for k, v in state_dict.items() if k in model_state_dict and v.shape != model_state_dict[k].shape]
            ignored_layers = [i for i in ignored_layers if any(x in i for x in ['head', 'criterion'])]         

            for k in ignored_layers: 
                if 'criterion' in k:
                    print('-', k, 'has been modified.')
                    print(state_dict[k].shape, '  ->  ', flush=True, end='')
                    state_dict[k] = torch.FloatTensor([conf["classes"][i][0] for i in conf["classes"]])
                    print(state_dict[k].shape)  
                else:
                    print('-', k, 'has been modified.')
                    print(state_dict[k].shape, '  ->  ', flush=True, end='')
                    state_dict[k] = 0 * np.abs(state_dict[k][0:num_classes])
                    print(state_dict[k].shape)
            
            seg_module.load_state_dict(state_dict, strict=False)
            
        print('###############################################################')
    else:
        print("Invalid checkpoint file path.")
        if exit_on_fail:
            raise SystemExit()
        print('###############################################################')
    print()
    

def training_stage(config, data_module, out_dir):
    """
    Conducts the training stage of the model: sets up the training environment, loads the model weights from a checkpoint if available,
    trains the model, and logs the training information.

    Parameters:
    config (dict): Configuration dictionary containing parameters for the task.
    data_module: Data module for training, validation, and testing.
    seg_module: Segmentation module for training.
    out_dir (Path): Path object representing the output directory.

    Returns:
    trained_state_dict (OrderedDict): The state dictionary of the trained model.
    """
    start = datetime.datetime.now()

    seed_everything(config['seed'], workers=True)

    seg_module = get_segmentation_module(config, stage='train')

    if config['tasks']['train_tasks']['init_weights_only_from_ckpt']:
        load_checkpoint(config, seg_module, exit_on_fail=False)

    ckpt_callback = train(config, data_module, seg_module, out_dir)

    best_trained_state_dict = torch.load(ckpt_callback.best_model_path, map_location=torch.device('cpu'))['state_dict']

    end = datetime.datetime.now()
    inference_time_seconds = end - start
    inference_time_seconds = inference_time_seconds.total_seconds()

    print(f"\n[Training finished in {str(timedelta(seconds=inference_time_seconds))} HH:MM:SS with {config['num_nodes']} nodes and {config['gpus_per_node']} gpus per node]") 
    print(f"Model path : {os.path.join(out_dir,'checkpoints')}\n\n")
    print('\n'+'-'*40)

    return best_trained_state_dict
   
   
def predict_stage(config, data_module, out_dir_predict, trained_state_dict=None):
    """
    Conducts the prediction stage of the model: sets up the prediction environment, loads the model weights from the training stage or a checkpoint file,
    and makes predictions.

    Parameters:
    config (dict): Configuration dictionary containing parameters for the task.
    data_module: Data module for training, validation, and testing.
    out_dir_predict (Path): Path object representing the output directory for predictions.
    trained_state_dict (OrderedDict, optional): The state dictionary of the trained model. Defaults to None.
    """
    seg_module = get_segmentation_module(config, stage='predict')
    if config['tasks']['train']:
        seg_module.load_state_dict(trained_state_dict, strict=False)  
    else:
        load_checkpoint(config, seg_module)
    predict(config, data_module, seg_module, out_dir_predict)


def main():
    # Read config and create output folder
    args = argParser.parse_args()
    config, out_dir = setup_environment(args)

    if config['codecarbon']:

        tracker = OfflineEmissionsTracker(output_dir=out_dir, 
                                project_name='flair-inc | codecarbon',
                                log_level="error",
                                country_iso_code='FRA'

        )
        tracker.start()

    # Custom Logger for console/logfile output
    sys.stdout = Logger(
        Path(config['paths']["out_folder"], config['paths']["out_model_name"], 'flair-compute.log').as_posix())
    print(datetime.datetime.now().strftime("Starting : %Y-%m-%d  %H:%M") + '\n')

    # Define data sets
    dict_train, dict_val, dict_test = get_datasets(config)
    print_recap(config, dict_train, dict_val, dict_test)

    # Copy relevant files for tracking
    if config["cp_csv_and_conf_to_output"]:
        copy_csv_and_config(config, out_dir, args)

    # Get LightningDataModule
    dm = get_data_module(config, dict_train=dict_train, dict_val=dict_val, dict_test=dict_test)

    # Initialize variable for weights
    trained_state_dict = None

    # Training
    if config['tasks']['train']:
        trained_state_dict = training_stage(config, dm, out_dir)

    # Inference
    if config['tasks']['predict']:
        out_dir_predict = Path(out_dir, 'results_'+config['paths']["out_model_name"])
        out_dir_predict.mkdir(parents=True, exist_ok=True)
        predict_stage(config, dm, out_dir_predict, trained_state_dict)


    if config['codecarbon']:

        tracker.stop()

        print("\n----- Emissions Tracking Summary -----")
        print(f"Total CO2 emissions: {tracker.final_emissions} kg CO2e")
        print(f"Total energy consumption: {tracker._total_energy} kWh")
        print(f"Total energy cpu: {tracker._total_cpu_energy} kWh")        
        print(f"Total energy gpu: {tracker._total_gpu_energy} kWh")
        print(f"Total energy ram: {tracker._total_ram_energy} kWh")
        print(f"Measures done : {tracker._measure_occurrence}.")


if __name__ == "__main__":
    main()
