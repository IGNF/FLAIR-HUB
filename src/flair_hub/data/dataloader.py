import numpy as np
import torch

from typing import Dict
from torch.utils.data import Dataset

from src.flair_hub.data.utils_data.io import read_patch
from src.flair_hub.data.utils_data.norm import norm
from src.flair_hub.data.utils_data.augmentations import apply_numpy_augmentations
from src.flair_hub.data.utils_data.label import reshape_label_ohe
from src.flair_hub.data.utils_data.elevation import calc_elevation
from src.flair_hub.data.utils_data.sentinel import (
    reshape_sentinel,
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
        
        # flair_dataset __init__
        if use_augmentations is True:
            self.use_augmentations = apply_numpy_augmentations
        else:
            self.use_augmentations = use_augmentations


        # Data and label setup (same as before)
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
                if mod == "SENTINEL2_TS":
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
                "data_paths": np.array(dict_paths[task]),
                "num_classes": len(label_conf["value_name"]),
                "channels": [label_conf.get("label_channel_nomenclature", 1)]
            }

    def _init_normalization(self):
        self.norm_type = self.config["modalities"]["normalization"]["norm_type"]
        enabled_modalities = self.config["modalities"]["inputs"]
        self.channels = {
            mod: self.config["modalities"]["inputs_channels"].get(mod, [])
            for mod, active in enabled_modalities.items() if active
        }
        self.normalization = {
            mod: {
                "mean": self.config["modalities"]["normalization"].get(f"{mod}_means", []),
                "std": self.config["modalities"]["normalization"].get(f"{mod}_stds", [])
            }
            for mod, active in enabled_modalities.items() if active
        }

    def __len__(self):
        for task in self.tasks.values():
            if len(task["data_paths"]) > 0:
                return len(task["data_paths"])
        return 0

    def __getitem__(self, index):
        batch = {}

        # Supervision
        for task, info in self.tasks.items():
            batch[f"ID_{task}"] = info["data_paths"][index]
            area_elem = info["data_paths"][index].split("/")[-1].split("_")
            area_elem = "_".join([area_elem[0], area_elem[-2], area_elem[-1].split(".")[0]])

        # AERIAL_RGBI
        key = "AERIAL_RGBI"
        if key in self.list_patch:
            data = read_patch(self.list_patch[key][index], self.channels[key])
            batch[key] = norm(data, 
                              self.norm_type, 
                              self.normalization[key]["mean"], 
                              self.normalization[key]["std"]
            )

        # AERIAL-RLT_PAN
        key = "AERIAL-RLT_PAN"
        if key in self.list_patch:
            data = read_patch(self.list_patch[key][index], self.channels[key])
            batch[key] = norm(data, 
                              self.norm_type, 
                              self.normalization[key]["mean"], 
                              self.normalization[key]["std"]
            )

        # DEM_ELEV
        key = "DEM_ELEV"
        if key in self.list_patch and self.list_patch[key][index] is not None:
            zdata = read_patch(self.list_patch[key][index])
            if self.config["modalities"]["pre_processings"]["calc_elevation"]:
                elev_data = calc_elevation(zdata)
                if self.config["modalities"]["pre_processings"]["calc_elevation_stack_dsm"]:
                    elev_data = np.stack((zdata[0, :, :], elev_data[0]), axis=0)
                batch[key] = elev_data
            else:
                batch[key] = zdata
            batch[key] = norm(
                batch[key],
                self.norm_type,
                self.normalization[key]["mean"],
                self.normalization[key]["std"]
            )

        # SPOT_RGBI
        key = "SPOT_RGBI"
        if key in self.list_patch:
            data = read_patch(self.list_patch[key][index], self.channels[key])
            batch[key] = norm(data, 
                              self.norm_type, 
                              self.normalization[key]["mean"], 
                              self.normalization[key]["std"]
            )

        # SENTINEL2_TS
        key = "SENTINEL2_TS"
        if key in self.list_patch:
            s2 = read_patch(self.list_patch[key][index])
            s2 = reshape_sentinel(s2, chunk_size=10)[:, [x - 1 for x in self.channels[key]], :, :]

            s2_dates_dict = self.dict_dates[key][area_elem]
            s2_dates = s2_dates_dict["dates"]
            s2_dates_diff = s2_dates_dict["diff_dates"]

            if self.config["modalities"]["pre_processings"]["filter_sentinel2"]:
                msk = read_patch(self.list_patch["SENTINEL2_MSK-SC"][index])
                msk = reshape_sentinel(msk, chunk_size=2)
                idx_valid = filter_time_series(
                    msk,
                    max_cloud_value=self.config["modalities"]["pre_processings"]["filter_sentinel2_max_cloud"],
                    max_snow_value=self.config["modalities"]["pre_processings"]["filter_sentinel2_max_snow"],
                    max_fraction_covered=self.config["modalities"]["pre_processings"]["filter_sentinel2_max_frac_cover"]
                )
                s2 = s2[np.where(idx_valid)[0]]
                s2_dates = s2_dates[np.where(idx_valid)[0]]
                s2_dates_diff = s2_dates_diff[np.where(idx_valid)[0]]

            if self.config["modalities"]["pre_processings"]["temporal_average_sentinel2"]:
                s2, s2_dates_diff = temporal_average(
                    s2, s2_dates,
                    period=self.config["modalities"]["pre_processings"]["temporal_average_sentinel2"],
                    ref_date=self.ref_date
                )

            batch[key] = s2
            batch[key.replace("_TS", "_DATES")] = s2_dates_diff

        # SENTINEL1 ASC
        key = "SENTINEL1-ASC_TS"
        if key in self.list_patch and self.list_patch[key][index] is not None:
            sentinel1asc_data = read_patch(self.list_patch[key][index])
            sentinel1asc_data = reshape_sentinel(sentinel1asc_data, chunk_size=2)[:, 
                [x - 1 for x in self.channels[key]], :, :]

            sentinel1asc_dates_dict = self.dict_dates[key][area_elem]
            sentinel1asc_dates = sentinel1asc_dates_dict['dates']
            sentinel1asc_dates_diff = sentinel1asc_dates_dict['diff_dates']

            if self.config["modalities"]["pre_processings"]["temporal_average_sentinel1"]:
                sentinel1asc_data, sentinel1asc_dates_diff = temporal_average(
                    sentinel1asc_data,
                    sentinel1asc_dates,
                    period=self.config["modalities"]["pre_processings"]["temporal_average_sentinel1"],
                    ref_date=self.ref_date
                )

            batch[key] = sentinel1asc_data
            batch[key.replace('_TS', '_DATES')] = sentinel1asc_dates_diff

        # SENTINEL1 DESC
        key = "SENTINEL1-DESC_TS"
        if key in self.list_patch and self.list_patch[key][index] is not None:
            sentinel1desc_data = read_patch(self.list_patch[key][index])
            sentinel1desc_data = reshape_sentinel(sentinel1desc_data, chunk_size=2)[:, 
                [x - 1 for x in self.channels[key]], :, :]

            sentinel1desc_dates_dict = self.dict_dates[key][area_elem]
            sentinel1desc_dates = sentinel1desc_dates_dict['dates']
            sentinel1desc_dates_diff = sentinel1desc_dates_dict['diff_dates']

            if self.config["modalities"]["pre_processings"]["temporal_average_sentinel1"]:
                sentinel1desc_data, sentinel1desc_dates_diff = temporal_average(
                    sentinel1desc_data,
                    sentinel1desc_dates,
                    period=self.config["modalities"]["pre_processings"]["temporal_average_sentinel1"],
                    ref_date=self.ref_date
                )

            batch[key] = sentinel1desc_data
            batch[key.replace('_TS', '_DATES')] = sentinel1desc_dates_diff

        # Labels
        for task, info in self.tasks.items():
            label = read_patch(info["data_paths"][index], info["channels"])
            batch[task] = reshape_label_ohe(label, info["num_classes"])

        # Apply numpy augmentations
        if callable(self.use_augmentations):
            input_keys = [k for k, v in self.config["modalities"]["inputs"].items() if v]
            label_keys = list(self.config["labels"])
            batch = self.use_augmentations(batch, input_keys, label_keys)

        # Convert to torch tensors
        batch = {
            k: torch.tensor(v, dtype=torch.float32)
            if isinstance(v, (np.ndarray, list)) and "ID_" not in k else v
            for k, v in batch.items()
        }

        return batch

