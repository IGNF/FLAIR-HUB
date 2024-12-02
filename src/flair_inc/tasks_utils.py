import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime


import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from typing import Dict, Any, Tuple, Optional

from flair_inc.models.flair_model import FLAIR_TimeTexture
from flair_inc.data_module import flair_datamodule
from flair_inc.task_module import SegmentationTask



    
def get_data_module(config, 
                    dict_train : dict =None, 
                    dict_val : dict = None, 
                    dict_test : dict = None,
                    ):
    """
    This function creates a data module for training, validation, and testing.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for the data module.
    dict_train (dict): Dictionary containing training data.
    dict_val (dict): Dictionary containing validation data.
    dict_test (dict): Dictionary containing test data.

    Returns:
    dm: Data module with specified configuration.
    """
    assert isinstance(config, dict), "config must be a dictionary"
    assert isinstance(config['modalities']["use_augmentation"], bool), "use_augmentation must be a boolean"
    assert isinstance(config['modalities']["use_metadata"], bool), "use_metadata must be a boolean"   
    
    if config['modalities']["use_augmentation"]:
        transform_set = A.Compose([A.VerticalFlip(p=0.5),
                                   A.HorizontalFlip(p=0.5),
                                   A.RandomRotate90(p=0.5)]
        )
    else:
        transform_set = None

    dm = flair_datamodule(
        config, 
        dict_train = dict_train,
        dict_val = dict_val,
        dict_test = dict_test,
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        drop_last = True,
        use_augmentations = transform_set,
    )
    
    return dm    



def get_segmentation_module(config, stage: str = 'train'):
    """
    This function creates a segmentation module for training or prediction.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for the segmentation module.
    stage (str): Stage for which the segmentation module is created ('train' or 'predict').

    Returns:
    seg_module: Segmentation module with specified configuration.
    """
    assert stage in ['train', 'predict'], "stage must be either 'train' or 'predict'"
                   
    # Define model
    model = FLAIR_TimeTexture(config)

    if stage == 'train':
        if config["use_weights"]:
            with torch.no_grad():
                class_weights = torch.FloatTensor([config["classes"][i][0] for i in config["classes"]])
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=4,
            min_lr=1e-7,
        )

        seg_module = SegmentationTask(
            model=model,
            config=config,
            class_infos=config["classes"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            use_metadata=config['modalities']["use_metadata"],
        )

    elif stage == 'predict':
        seg_module = SegmentationTask(
            model=model,
            config=config,
            class_infos=config["classes"],
            use_metadata=config['modalities']["use_metadata"],
        )        

    return seg_module  





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

    if config['modalities']['use_metadata']:
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
