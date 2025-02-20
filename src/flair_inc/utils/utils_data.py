import os
import sys
import pandas as pd
import geopandas as gpd
import json
import datetime
import numpy as np
import numpy as np
import torch

from skimage import img_as_float
from torch.nn import functional as F
from typing import Dict, Any, Tuple, Optional
from pytorch_lightning.utilities.rank_zero import rank_zero_only 



def get_paths(config: Dict[str, Any], split: str = 'train') -> Dict:
    """
    Retrieves paths to data files based on the provided configuration and split type.
    Args:
        config (dict): A configuration dictionary that includes paths to CSV files,metadata usage, and modality activation.
        split (str): The data split type, which can be 'train', 'val', or 'test'.
    Returns:
        dict: A dictionary containing paths for each modality and metadata if specified.
    Raises:
        SystemExit: If an invalid split is specified or the CSV file path is invalid.
    """
    
    if split == 'train':
        csv_path = config['paths']['train_csv']
    elif split == 'val':
        csv_path = config['paths']['val_csv']
    elif split == 'test':
        csv_path = config['paths']['test_csv']
    else:
        print("Invalid split specified.")
        raise SystemExit()

    if csv_path is not None and os.path.isfile(csv_path) and csv_path.endswith('.csv'):
        paths = pd.read_csv(csv_path)
    else:
        print(f"Invalid .csv file path for {split} split.")
        raise SystemExit()

    dict_paths = {modality: [] for modality in config['modalities']['inputs'].keys()}

    for modality, is_active in config['modalities']['inputs'].items():
        if is_active == True and modality in paths.columns:
            dict_paths[modality] = paths[modality].tolist()


    for label_mod in config['labels']:
        dict_paths[label_mod] = paths[label_mod].tolist()

    if config['modalities']['inputs']['SENTINEL2_TS']:
        dict_paths['SENTINEL2_MSK-SC'] = paths['SENTINEL2_MSK-SC'].tolist()
    else:
        dict_paths['SENTINEL2_MSK-SC'] = []
        
    return dict_paths


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


def prepare_sentinel_dates(config: Dict[str, Any], file_path: str) -> Dict:
    gdf = gpd.read_file(file_path)
    ref_month, ref_day = map(int, config['models']['multitemp_model']['ref_date'].split('-'))

    dict_dates = {}
    for _, row in gdf.iterrows():
        area_id = row['zone_id']
        acquisition_dates = json.loads(row['acquisition_dates'])
        
        dates_array = []
        diff_dates_array = []        
        for date_str in acquisition_dates.values():
            try:
                original_date = datetime.datetime.strptime(date_str, "%Y%m%d")
                reference_date = datetime.datetime(original_date.year, ref_month, ref_day)
                diff_days = (original_date - reference_date).days
                dates_array.append(original_date)
                diff_dates_array.append(diff_days)
            except ValueError as e:
                print(f"Invalid date encountered: {date_str}. Error: {e}")

        dict_dates[area_id] = {
            'dates': np.array(dates_array),
            'diff_dates': np.array(diff_dates_array)
        }
    return dict_dates


def get_sentinel_dates_mtd(config: dict) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Retrieve sentinel dates metadata based on the provided configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        tuple: Dictionaries with area_id as keys and acquisition_dates as values for Sentinel2, Sentinel1-ASC, and Sentinel1-DESC.
    """
    assert isinstance(config, dict), "config must be a dictionary"

    dates_s2, dates_s1asc, dates_s1desc = {}, {}, {}

    sen2_used = config['modalities']['inputs'].get('SENTINEL2_TS', False)
    sen1asc_used = config['modalities']['inputs'].get('SENTINEL1-ASC_TS', False)
    sen1desc_used = config['modalities']['inputs'].get('SENTINEL1-DESC_TS', False)

    if not (sen2_used or sen1asc_used or sen1desc_used):
        return dates_s2, dates_s1asc, dates_s1desc

    if sen2_used:
        dates_s2 = prepare_sentinel_dates(config, config['paths']['global_mtd_folder'] + 'GLOBAL_SENTINEL2_MTD_DATES.gpkg')
    if sen1asc_used:
        dates_s1asc = prepare_sentinel_dates(config, config['paths']['global_mtd_folder'] + 'GLOBAL_SENTINEL1-ASC_MTD_DATES.gpkg')
    if sen1desc_used:
        dates_s1desc = prepare_sentinel_dates(config, config['paths']['global_mtd_folder'] + 'GLOBAL_SENTINEL1-DESC_MTD_DATES.gpkg')

    return dates_s2, dates_s1asc, dates_s1desc


def norm(
    in_img: np.ndarray, 
    norm_type: str = None, 
    means: list[float] = [], 
    stds: list[float] = []
) -> np.ndarray:
    """
    Normalize an image array using different normalization strategies.
    Args:
        in_img (np.ndarray): Input image array to be normalized. 
            It should have a shape where the first dimension corresponds to channels.
        norm_type (str, optional): Normalization type, either 'scaling', 'custom', or 'without'.
            - 'scaling': Scales the image to [0, 1] using `skimage.util.img_as_float`.
            - 'custom': Normalizes each channel using provided means and standard deviations.
            - 'without': No normalization is applied.
        means (list[float], optional): List of means for each channel (used for 'custom' normalization).
        stds (list[float], optional): List of standard deviations for each channel 
            (used for 'custom' normalization).
    Returns:
        np.ndarray: Normalized image array.
    Exits:
        If an invalid `norm_type` is provided or `means` and `stds` lengths mismatch when 
        using 'custom', the program exits with an error message.
    """
    try:
        if norm_type not in ['scaling', 'custom', 'without']:
            print("Error: Normalization argument should be 'scaling', 'custom', or 'without'.")
            sys.exit(1)
        
        if norm_type == 'custom':
            if len(means) != len(stds):
                print("Error: If using 'custom', the provided means and stds must have the same length.")
                sys.exit(1)
            in_img = in_img.astype(np.float64)
            for i in range(in_img.shape[0]):  # Assuming first dimension is channels
                in_img[i] -= means[i]
                in_img[i] /= stds[i]
        elif norm_type == 'scaling':
            in_img = img_as_float(in_img)
    
        return in_img
    
    except Exception as e:
        print(f"Unexpected error during normalization: {e}")
        sys.exit(1)


def _calculate_padding_length(x: np.ndarray, l: int) -> int:
    """
    Calculate the required padding length based on the target length.
    Parameters:
    - x (np.ndarray): The input tensor to be padded.
    - l (int): The target length for the first dimension (axis 0).
    Returns:
    - int: The padding length for the first dimension (axis 0).
    """
    padlen = l - x.shape[0]
    return padlen


def _create_padding_array(x: np.ndarray, padlen: int) -> list:
    """
    Creates a padding array to be used with `F.pad`.
    Parameters:
    - x (np.ndarray): The input tensor to be padded.
    - padlen (int): The padding length for the first dimension (axis 0).
    Returns:
    - list: The padding array to use with `F.pad`.
    """
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return pad


def pad_tensor(x: np.ndarray, l: int, pad_value: int = 0) -> np.ndarray:
    """
    Pads the tensor `x` along the first dimension to the target length `l`.
    Parameters:
    - x (np.ndarray): The input tensor to be padded.
    - l (int): The target length for the first dimension (axis 0).
    - pad_value (int, optional): The value to pad with. Default is 0.
    Returns:
    - np.ndarray: The padded tensor.
    """
    padlen = _calculate_padding_length(x, l)
    pad = _create_padding_array(x, padlen)
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate_flair(sample_dict, pad_value=0):
    """
    Collate function for batching and padding. 
    Pads only the relevant SENTINEL fields and keeps other fields unchanged.
    Args:
    - sample_dict (list): List of dictionaries where each dictionary represents a sample.
    - pad_value (int): The value used for padding tensors.
    Returns:
    - dict: A dictionary with padded tensors and original string keys.
    """

    TO_PAD_KEYS = [
        'SENTINEL2_TS', 'SENTINEL2_DATES',
        'SENTINEL1-ASC_TS', 'SENTINEL1-ASC_DATES',
        'SENTINEL1-DESC_TS', 'SENTINEL1-DESC_DATES'
    ]
    
    batch = {}
    
    for key in sample_dict[0].keys():
        if key in TO_PAD_KEYS:
            data = [i[key] for i in sample_dict]

            if all(len(e) == 0 for e in data):
                batch[key] = torch.empty((len(data), 0))  
                continue

            sizes = [e.shape[0] for e in data if len(e) > 0]
            max_size = max(sizes) if sizes else 0

            padded_data = [
                pad_tensor(d, max_size, pad_value=pad_value) if len(d) > 0 else torch.zeros((max_size,), dtype=d.dtype) 
                for d in data
            ]
            batch[key] = torch.stack(padded_data, dim=0)

        elif isinstance(sample_dict[0][key], torch.Tensor):
            batch[key] = torch.stack([i[key] for i in sample_dict], dim=0)
        else:
            batch[key] = [i[key] for i in sample_dict]

    return batch


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