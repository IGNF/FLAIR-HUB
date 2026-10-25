import os
from pathlib import Path
import pandas as pd

from typing import Dict, Any, Tuple, Optional, List, Set

from flair_hub.data.utils_data.sentinel_dates import get_sentinel_dates_mtd


def extract_sentinel_patch_ids(dicts: List[Dict]) -> Set[str]:
    """
    Extracts unique patch IDs from Sentinel file paths in a list of dictionaries.
    Looks for keys: 'SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS' and
    parses patch IDs from filenames.
    Args:
        dicts (List[Dict]): List of dictionaries with optional Sentinel file paths.
    Returns:
        Set[str]: Set of unique patch IDs.
    """
    patch_ids = set()
    for d in dicts:
        if d is None:
            continue
        for key in ['SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS']:
            for path in d.get(key, []):
                patch_id = Path(path).stem.replace(f'_{key}', '')
                patch_ids.add(patch_id)
    return patch_ids


def get_paths(config: Dict[str, Any], split: str = 'train') -> Dict:
    """
    Retrieves paths to data files based on the provided configuration and split type.
    Args:
        config (dict): A configuration dictionary that includes paths to CSV files, and modality activation.
        split (str): The data split type, which can be 'train', 'val', or 'test'.
    Returns:
        dict: A dictionary containing paths for each modality.
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
        if is_active:
            if modality not in paths or paths[modality].isna().any():
                raise ValueError(f'{split}: missing paths for active modality {modality}')
            dict_paths[modality] = paths[modality].tolist()

    for label_mod in config['labels']:
        dict_paths[label_mod] = [None if pd.isna(value) else value for value in
                                 paths.get(label_mod, pd.Series([None] * len(paths)))]
    if split != 'test' and any(
        all(dict_paths[task][index] is None for task in config['labels'])
        for index in range(len(paths))
    ):
        raise ValueError(f'{split}: every sample requires at least one annotated task')
    if split != 'test':
        for task in config['labels']:
            if not any(path is not None for path in dict_paths[task]):
                raise ValueError(f'{split}: active task {task} has no annotations')

    if (config['modalities']['inputs']['SENTINEL2_TS'] and
            config['modalities']['pre_processings']['filter_sentinel2']):
        dict_paths['SENTINEL2_MSK-SC'] = paths['SENTINEL2_MSK-SC'].tolist()
    else:
        dict_paths['SENTINEL2_MSK-SC'] = []
    for key, values in dict_paths.items():
        missing = [path for path in set(values) if path is not None
                   and not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}/{key}: {len(missing)} missing rasters, e.g. {missing[:3]}')
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
    if not config['tasks']['train'] and not config['tasks']['predict']:
        return dict_train, dict_val, dict_test

    if config['tasks']['train']:
        dict_train = get_paths(config, split='train')
        dict_val   = get_paths(config, split='val')

    if config['tasks']['predict']:
        dict_test  = get_paths(config, split='test')

    all_dicts = [dict_train, dict_val, dict_test]
    used_patch_ids = extract_sentinel_patch_ids(all_dicts)

    dates_s2, dates_s1asc, dates_s1desc = get_sentinel_dates_mtd(config, used_patch_ids)
    for modality, dates in (("SENTINEL2_TS", dates_s2),
                            ("SENTINEL1-ASC_TS", dates_s1asc),
                            ("SENTINEL1-DESC_TS", dates_s1desc)):
        if not config["modalities"]["inputs"].get(modality):
            continue
        expected = {Path(path).stem.replace("_" + modality, "")
                    for split in all_dicts if split for path in split.get(modality, [])}
        missing = expected - set(dates)
        empty = [key for key in expected & set(dates) if not len(dates[key]["dates"])]
        if missing or empty:
            raise ValueError(f"{modality}: missing/empty acquisition dates for {sorted(missing)[:5] + empty[:5]}")

    for d in [dict_train, dict_val, dict_test]:
        if d is not None:
            d['DATES_S2'] = dates_s2
            d['DATES_S1_ASC'] = dates_s1asc
            d['DATES_S1_DESC'] = dates_s1desc

    return dict_train, dict_val, dict_test
