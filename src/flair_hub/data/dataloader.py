from flair_hub.data.utils_data.sentinel import temporal_band_indexes
import numpy as np
import torch
from pathlib import Path

from typing import Dict
from torch.utils.data import Dataset

from flair_hub.data.utils_data.io import read_patch
from flair_hub.utils.class_config import label_lookup, encode_labels
from flair_hub.data.utils_data.augmentations import (
    apply_configured_augmentations,
    pad_sample_to_original_sizes,
)
from flair_hub.data.utils_data.preprocessing import prepare_model_input, reorder_bands
from flair_hub.data.utils_data.elevation import calc_elevation
from flair_hub.data.utils_data.sentinel import (
                                                reshape_sentinel, prepare_time_series,
                                                filter_time_series,
                                                temporal_average,
)


class flair_dataset(Dataset):
    """
    PyTorch Dataset for multimodal remote sensing data.
    Applies normalization, temporal aggregation, cloud filtering, and augmentations.
    Args:
        config (Dict): Configuration dictionary containing model and modality-specific settings, such as
            input channels, normalization parameters, and temporal processing options.
        dict_paths (Dict): A dictionary mapping data modalities (e.g., "AERIAL_RGBI", "LABELS")
            to their corresponding file paths.
        use_augmentations (callable, optional): A callable function or transformation pipeline to
            apply augmentations to the samples. Defaults to None.
     Methods:
        __len__() -> int:
            Returns the number of samples in the dataset.
        __getitem__(index: int) -> Dict:
            Retrieves batch_elements. Applies
            normalization, temporal aggregation, and augmentations (if specified).

    """

    def __init__(self, config: Dict, dict_paths: Dict, use_augmentations: callable = None) -> None:

        self.config = config
        if use_augmentations is True:
            self.use_augmentations = apply_configured_augmentations
        else:
            self.use_augmentations = use_augmentations


        self._init_data_paths(dict_paths)
        self._init_label_info(dict_paths)
        self._init_normalization()
        self.ref_date = config['models']['multitemp_model']['ref_date']


    def _init_data_paths(self, dict_paths):
        self.list_patch = {}
        enabled = self.config["modalities"]["inputs"]
        for mod, enabled_flag in enabled.items():
            if enabled_flag and mod in dict_paths:
                self.list_patch[mod] = np.array(dict_paths[mod])
                if mod == "SENTINEL2_TS" and self.config["modalities"]["pre_processings"]["filter_sentinel2"]:
                    self.list_patch["SENTINEL2_MSK-SC"] = np.array(dict_paths["SENTINEL2_MSK-SC"])

        self.dict_dates = {}
        if "SENTINEL2_TS" in enabled:
            self.dict_dates["SENTINEL2_TS"] = dict_paths.get("DATES_S2", {})
        if "SENTINEL1-ASC_TS" in enabled:
            self.dict_dates["SENTINEL1-ASC_TS"] = dict_paths.get("DATES_S1_ASC", {})
        if "SENTINEL1-DESC_TS" in enabled:
            self.dict_dates["SENTINEL1-DESC_TS"] = dict_paths.get("DATES_S1_DESC", {})

    def _init_label_info(self, dict_paths):
        self.tasks = {}
        for task in self.config["labels"]:
            label_conf = self.config["labels_configs"][task]
            self.tasks[task] = {
                "label_lookup": label_lookup(label_conf),
                "data_paths": np.array(dict_paths.get(task, []), dtype=object),
                "channels": [label_conf.get("label_channel_nomenclature", 1)],
            }

    def _init_normalization(self):
        normalization_cfg = self.config["modalities"]["normalization"]
        default_norm_type = normalization_cfg["norm_type"]
        enabled_modalities = self.config["modalities"]["inputs"]
        self.channels = {
            mod: self.config["modalities"]["inputs_channels"].get(mod, [])
            for mod, active in enabled_modalities.items() if active
        }
        self.read_channels = {
            mod: self.config["modalities"].get("read_channels", {}).get(
                mod, self.channels.get(mod, [])
            )
            for mod, active in enabled_modalities.items() if active
        }
        self.normalization = {
            mod: {
                "type": normalization_cfg.get("types", {}).get(mod, default_norm_type),
                "mean": normalization_cfg.get(f"{mod}_means", []),
                "std": normalization_cfg.get(f"{mod}_stds", []),
                "scale": normalization_cfg.get(f"{mod}_scale", 1.0),
                "offset": normalization_cfg.get(f"{mod}_offset", 0.0)
            }
            for mod, active in enabled_modalities.items() if active
        }

    def __len__(self):
        return len(next(iter(self.list_patch.values())))

    def __getitem__(self, index):
        batch = {}
        reference_mod = next(iter(self.list_patch))
        reference_path = self.list_patch[reference_mod][index]
        temporal_mod = next((m for m in self.list_patch if m.endswith('_TS')), None)
        if temporal_mod:
            area_elem = Path(self.list_patch[temporal_mod][index]).stem.replace('_' + temporal_mod, '')
        for task, info in self.tasks.items():
            path = info['data_paths'][index]
            batch[f'ID_{task}'] = path or reference_path
            batch[f'HAS_LABEL_{task}'] = path is not None


        key = "AERIAL_RGBI"
        if key in self.list_patch:
            batch[key] = read_patch(
                self.list_patch[key][index], self.read_channels[key]
            ).astype(np.float32)


        key = "AERIAL-RLT_PAN"
        if key in self.list_patch:
            batch[key] = read_patch(
                self.list_patch[key][index], self.read_channels[key]
            ).astype(np.float32)


        key = "DEM_ELEV"
        if key in self.list_patch and self.list_patch[key][index] is not None:
            zdata = read_patch(self.list_patch[key][index], self.read_channels[key]).astype(np.float32)
            zdata = reorder_bands(zdata, self.read_channels[key], self.channels[key])
            if self.config["modalities"]["pre_processings"]["calc_elevation"]:
                elev_data = calc_elevation(zdata)
                if self.config["modalities"]["pre_processings"]["calc_elevation_stack_dsm"]:
                    elev_data = np.stack((zdata[0, :, :], elev_data[0]), axis=0)
                batch[key] = elev_data
            else:
                batch[key] = zdata


        key = "SPOT_RGBI"
        if key in self.list_patch:
            batch[key] = read_patch(
                self.list_patch[key][index], self.read_channels[key]
            ).astype(np.float32)


        preprocessing = self.config['modalities']['pre_processings']
        for key in ('SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS'):
            if key not in self.list_patch:
                continue
            date_info = self.dict_dates[key][area_elem]
            raw = read_patch(self.list_patch[key][index])
            indexes = temporal_band_indexes(raw.shape[0], len(date_info['dates']), self.channels[key])
            series = reshape_sentinel(raw[[i - 1 for i in indexes]], len(self.channels[key]))
            mask = None
            if key == 'SENTINEL2_TS' and preprocessing['filter_sentinel2']:
                mask = reshape_sentinel(read_patch(self.list_patch['SENTINEL2_MSK-SC'][index]), 2)
            series, dates = prepare_time_series(
                series, date_info['dates'], date_info['diff_dates'], mask=mask,
                max_cloud=preprocessing['filter_sentinel2_max_cloud'],
                max_snow=preprocessing['filter_sentinel2_max_snow'],
                maximum_covered_fraction=preprocessing['filter_sentinel2_max_frac_cover'],
                period=preprocessing['temporal_average_sentinel2' if key == 'SENTINEL2_TS'
                                     else 'temporal_average_sentinel1'], ref_date=self.ref_date)
            batch[key] = series
            batch[key.replace('_TS', '_DATES')] = dates


        finest = max((m for m in self.list_patch if m in batch),
                     key=lambda m: batch[m].shape[-2] * batch[m].shape[-1])
        for task, info in self.tasks.items():
            path = info['data_paths'][index]
            if path is None:
                batch[f'ID_{task}'] = self.list_patch[finest][index]
                batch[task] = np.full((1, *batch[finest].shape[-2:]), -100, dtype=np.int64)
            else:
                label = read_patch(path, info['channels'])
                batch[task] = encode_labels(label, info['label_lookup'])

        input_keys = [
            key
            for key, enabled in self.config["modalities"]["inputs"].items()
            if enabled
        ]
        label_keys = list(self.config["labels"])

        for key in input_keys:
            if key not in batch or not isinstance(batch[key], np.ndarray):
                continue
            loaded_order = (
                self.channels[key]
                if batch[key].ndim == 4 or key == "DEM_ELEV"
                else self.read_channels.get(key, self.channels[key])
            )
            batch[key] = reorder_bands(
                batch[key], loaded_order, self.channels[key]
            )

        target_sizes = {
            key: batch[key].shape[-2:]
            for key in input_keys + label_keys
            if key in batch and isinstance(batch[key], np.ndarray)
            and batch[key].ndim in (3, 4)
        }
        empty_temporal_frames = {
            key: np.all(batch[key] == 0, axis=(1, 2, 3))
            for key in input_keys if key in batch and batch[key].ndim == 4
        }

        if callable(self.use_augmentations):
            batch = self.use_augmentations(
                batch,
                input_keys,
                label_keys,
                self.config.get("augmentation", {}),
            )

        for key in input_keys:
            if key not in batch:
                continue
            batch[key] = prepare_model_input(
                batch[key],
                loaded_band_order=self.channels[key],
                input_band_order=self.channels[key],
                normalization_type=self.normalization[key]["type"],
                mean=self.normalization[key]["mean"],
                std=self.normalization[key]["std"],
                scale=self.normalization[key]["scale"],
                offset=self.normalization[key]["offset"],
            )
            if key in empty_temporal_frames:
                batch[key][empty_temporal_frames[key]] = 0

        batch = pad_sample_to_original_sizes(batch, target_sizes, label_keys)


        batch = {
            key: torch.tensor(
                value,
                dtype=torch.long if key in self.tasks else torch.float32,
            )
            if isinstance(value, (np.ndarray, list)) and "ID_" not in key
            else value
            for key, value in batch.items()
        }

        return batch

