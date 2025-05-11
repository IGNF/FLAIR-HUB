import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from datetime import datetime
import pandas as pd

from src.flair_hub.data.utils_data.norm import norm as normalize_array
from src.flair_hub.data.utils_data.sentinel import (
    reshape_sentinel,
    filter_time_series,
    temporal_average
)


class MultiModalSlicedDataset(Dataset):
    def __init__(self, dataframe, modality_cfgs, patch_size_dict, ref_date_str, modalities_config):
        self.df = dataframe
        self.modalities = modality_cfgs
        self.modalities_config = modalities_config
        self.patch_sizes = patch_size_dict
        self.ref_date_str = ref_date_str
        self.readers = {
            mod: rasterio.open(cfg['input_img_path']) for mod, cfg in modality_cfgs.items()
        }

        self.mask_reader = None
        self.mask_resolution_ratio = 1.0

        sentinel_cfg = modality_cfgs.get("SENTINEL2_TS")
        if sentinel_cfg and sentinel_cfg.get("filter_clouds") and "filter_clouds_img_path" in sentinel_cfg:
            self.mask_reader = rasterio.open(sentinel_cfg["filter_clouds_img_path"])
            sentinel_res = self.readers["SENTINEL2_TS"].res[0]
            mask_res = self.mask_reader.res[0]
            self.mask_resolution_ratio = sentinel_res / mask_res

        self.diff_dates = self._init_diff_dates()

    def _init_diff_dates(self):
        diff_dates = {}
        ref_month, ref_day = map(int, self.ref_date_str.split('-'))
        dummy_year = 2025
        ref_date = datetime(dummy_year, ref_month, ref_day)

        for mod, cfg in self.modalities.items():
            if not mod.endswith("_TS"):
                continue

            # If cloud filtering is required, dates_txt must be present and valid
            if cfg.get("filter_clouds", False):
                if "dates_txt" not in cfg or not cfg["dates_txt"]:
                    raise ValueError(f"[✗] 'filter_clouds' is enabled for '{mod}' but 'dates_txt' is missing or empty.")

            # Only proceed if dates_txt is actually defined and used
            if 'dates_txt' in cfg and cfg['dates_txt']:
                with open(cfg['dates_txt'], 'r') as f:
                    date_strs = [line.strip() for line in f if line.strip()]
                if not date_strs:
                    raise ValueError(f"[✗] 'dates_txt' file for '{mod}' is empty.")
                dates = [datetime.strptime(d, '%Y%m%d') for d in date_strs]
                date_diffs = [(d - datetime(d.year, ref_date.month, ref_date.day)).days for d in dates]
                diff_dates[mod] = {
                    'dates': pd.Series(dates),
                    'diff_dates': np.array(date_diffs)
                }

        return diff_dates



    def _load_patch(self, reader, bounds, cfg, patch_size, mod_name=None):
        window = from_bounds(*bounds, transform=reader.transform)

        if mod_name and mod_name.endswith("_TS") and mod_name in self.diff_dates:
            num_dates = len(self.diff_dates[mod_name]['dates'])
            num_channels = len(cfg['channels'])
            total_bands = num_channels * num_dates
            indexes = list(range(1, total_bands + 1))
        else:
            indexes = cfg['channels']

        patch = reader.read(
            indexes=indexes,
            window=window,
            out_shape=(len(indexes), patch_size, patch_size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0
        )
        return patch, window


    def _normalize_patch(self, patch, cfg):
        norm_cfg = cfg.get('normalization', {})
        if norm_cfg:
            return normalize_array(patch, norm_cfg.get("type"), norm_cfg.get("means"), norm_cfg.get("stds"))
        return patch

    def _process_time_series_patch(self, mod_name, patch, window, cfg):
        patch = reshape_sentinel(patch, chunk_size=len(cfg['channels']))

        if mod_name == "SENTINEL2_TS" and self.mask_reader:

            # Number of timestamps = number of date entries
            num_timestamps = len(self.diff_dates[mod_name]['dates'])

            # Read all 2*T bands
            num_bands = 2 * num_timestamps
            h = int(patch.shape[2] / self.mask_resolution_ratio)
            w = int(patch.shape[3] / self.mask_resolution_ratio)

            msk = self.mask_reader.read(
                indexes=list(range(1, num_bands + 1)),  # 1-based indexing
                window=window,
                out_shape=(num_bands, h, w),
                resampling=Resampling.nearest,
                boundless=True,
                fill_value=0
            )

            msk = reshape_sentinel(msk, chunk_size=2)
            valid_idx = filter_time_series(msk)

            patch = patch[valid_idx]
            self.diff_dates[mod_name]['dates'] = self.diff_dates[mod_name]['dates'][valid_idx]
            self.diff_dates[mod_name]['diff_dates'] = self.diff_dates[mod_name]['diff_dates'][valid_idx]

        if cfg.get('temporal_average', False):
            patch, diffs = temporal_average(
                patch,
                self.diff_dates[mod_name]['dates'],
                period=cfg.get("average_period", "monthly"),
                ref_date=self.ref_date_str
            )
            self.diff_dates[mod_name]['diff_dates'] = diffs

        return patch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bounds = row.geometry.bounds
        tile_data = {}

        for mod_name, cfg in self.modalities.items():
            reader = self.readers[mod_name]
            patch_size = self.patch_sizes[mod_name]
            patch, window = self._load_patch(reader, bounds, cfg, patch_size, mod_name)

            if mod_name.endswith("_TS"):
                patch = self._process_time_series_patch(mod_name, patch, window, cfg)
                tile_data[mod_name] = torch.tensor(patch, dtype=torch.float32)
                tile_data[mod_name.replace('_TS', '_DATES')] = torch.tensor(
                    self.diff_dates[mod_name]['diff_dates'], dtype=torch.float32)
            else:
                raw_patch = patch.copy()
                patch = self._normalize_patch(patch, cfg)
                tile_data[mod_name] = torch.tensor(patch, dtype=torch.float32)
                tile_data[mod_name + '_RAW'] = torch.tensor(raw_patch, dtype=torch.float32)

        tile_data['index'] = torch.tensor([idx], dtype=torch.long)

        for task in self.modalities_config["labels"]:
            num_classes = len(self.modalities_config["labels_configs"][task]["value_name"])
            patch_size = list(self.patch_sizes.values())[0]  # Use reference patch size
            dummy_label = torch.zeros((num_classes, patch_size, patch_size), dtype=torch.float32)
            tile_data[task] = dummy_label

        return tile_data

    def __del__(self):
        for reader in self.readers.values():
            reader.close()
        if self.mask_reader:
            self.mask_reader.close()
