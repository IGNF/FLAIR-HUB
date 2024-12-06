import os
import pandas as pd
import geopandas as gpd
import json
import datetime
import numpy as np
import numpy as np
import torch

from torch.nn import functional as F
from typing import Dict, Any, Tuple, Optional



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
    dict_paths['LABELS'] = paths[config['modalities']['labels']].tolist()

    if config['modalities']['inputs']['SENTINEL2_TS']:
        dict_paths['SENTINEL2_MSK-SC'] = paths['SENTINEL2_MSK-SC'].tolist()
    else:
        dict_paths['SENTINEL2_MSK-SC'] = []

    if config['modalities']['pre_processings']['use_metadata']:
        dict_paths['MTD'] = parsing_metadata(dict_paths.get('AERIAL_RGBI', []), config)
    else:
        dict_paths['MTD'] = []
        
    return dict_paths


def prepare_sentinel_dates(config: Dict[str, Any], file_path: str) -> Dict:
    gdf = gpd.read_file(file_path)
    ref_month, ref_day = map(int, config['models']['multitemp_model']['ref_date'].split('-'))

    dict_dates = {}
    for _, row in gdf.iterrows():
        area_id = row['area_id']
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


def parsing_metadata(image_path_list, config):
    #### encode metadata
    def coordenc_opt(coords, enc_size=32) -> np.array:
        d = int(enc_size/2)
        d_i = np.arange(0, d / 2)
        freq = 1 / (10e7 ** (2 * d_i / d))

        x,y = coords[0]/10e7, coords[1]/10e7
        enc = np.zeros(d * 2)
        enc[0:d:2]    = np.sin(x * freq)
        enc[1:d:2]    = np.cos(x * freq)
        enc[d::2]     = np.sin(y * freq)
        enc[d + 1::2] = np.cos(y * freq)
        return list(enc)           

    def norm_alti(alti: int) -> float:
        min_alti = 0
        max_alti = 3164.9099121094
        return [(alti-min_alti) / (max_alti-min_alti)]        

    def format_cam(cam: str) -> np.array:
        return [[1,0] if 'UCE' in cam else [0,1]][0]

    def cyclical_enc_datetime(date: str, time: str) -> list:
        def norm(num: float) -> float:
            return (num-(-1))/(1-(-1))
        year, month, day = date.split('-')
        if year == '2018':   enc_y = [1,0,0,0]
        elif year == '2019': enc_y = [0,1,0,0]
        elif year == '2020': enc_y = [0,0,1,0]
        elif year == '2021': enc_y = [0,0,0,1]    
        sin_month = np.sin(2*np.pi*(int(month)-1/12)) ## months of year
        cos_month = np.cos(2*np.pi*(int(month)-1/12))    
        sin_day = np.sin(2*np.pi*(int(day)/31)) ## max days
        cos_day = np.cos(2*np.pi*(int(day)/31))     
        h,m=time.split('h')
        sec_day = int(h) * 3600 + int(m) * 60
        sin_time = np.sin(2*np.pi*(sec_day/86400)) ## total sec in day
        cos_time = np.cos(2*np.pi*(sec_day/86400))
        return enc_y+[norm(sin_month),norm(cos_month),norm(sin_day),norm(cos_day),norm(sin_time),norm(cos_time)]     
    
    
    with open(config['paths']['path_metadata_aerial'], 'r') as f:
        metadata_dict = json.load(f)  
    
    MTD = []
    for img in image_path_list:
        curr_img     = img.split('/')[-1][:-4]
        enc_coords   = coordenc_opt([metadata_dict[curr_img]["patch_centroid_x"], metadata_dict[curr_img]["patch_centroid_y"]])
        enc_alti     = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
        enc_camera   = format_cam(metadata_dict[curr_img]['camera'])
        enc_temporal = cyclical_enc_datetime(metadata_dict[curr_img]['date'], metadata_dict[curr_img]['time'])
        mtd_enc      = enc_coords+enc_alti+enc_camera+enc_temporal 
        MTD.append(mtd_enc)  
        
    return MTD


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
        'SENTINEL2_TS', 'SENTINEL2_TS_DATES',
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
