from flair_hub.data.utils_data.sentinel import temporal_band_indexes
import os
import numpy as np
import torch
import rasterio
import pandas as pd

from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from rasterio.io import DatasetReader
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

from flair_hub.data.utils_data.preprocessing import prepare_model_input, reorder_bands
from flair_hub.data.utils_data.elevation import calc_elevation
from flair_hub.data.utils_data.sentinel import (
                                                reshape_sentinel, prepare_time_series,
                                                filter_time_series,
                                                temporal_average
)
class MultiModalSlicedDataset(Dataset):
    """
    Dataset for loading geospatial patches across multiple modalities (mono- and multi-temporal).
    Handles time series reshaping, cloud masking, and temporal averaging.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        modality_cfgs: Dict[str, Dict[str, Any]],
        patch_size_dict: Dict[str, int],
        ref_date_str: str,
        modalities_config: Dict[str, Any]
    ) -> None:
        self.df = dataframe
        self.modalities = modality_cfgs
        self.modalities_config = modalities_config
        self.patch_sizes = patch_size_dict
        self.ref_date_str = ref_date_str

        self.readers: Dict[str, DatasetReader] = {
            mod: rasterio.open(cfg['input_img_path']) for mod, cfg in modality_cfgs.items()
        }

        self.mask_reader: Optional[DatasetReader] = None
        self.mask_resolution_ratio: float = 1.0

        sentinel_cfg = modality_cfgs.get("SENTINEL2_TS")
        if sentinel_cfg and sentinel_cfg.get("filter_clouds") and "filter_clouds_img_path" in sentinel_cfg:
            self.mask_reader = rasterio.open(sentinel_cfg["filter_clouds_img_path"])
            if self.mask_reader.crs != self.readers["SENTINEL2_TS"].crs:
                raise ValueError("Sentinel image and cloud mask must use the same CRS")
            sentinel_res = self.readers["SENTINEL2_TS"].res[0]
            mask_res = self.mask_reader.res[0]
            self.mask_resolution_ratio = sentinel_res / mask_res

        self.diff_dates = self._init_diff_dates()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["readers"] = {}
        state["mask_reader"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.readers = {mod: rasterio.open(cfg["input_img_path"])
                        for mod, cfg in self.modalities.items()}
        sentinel = self.modalities.get("SENTINEL2_TS", {})
        if sentinel.get("filter_clouds"):
            self.mask_reader = rasterio.open(sentinel["filter_clouds_img_path"])

    def _init_diff_dates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize date differences for time series modalities."""
        diff_dates = {}
        ref_month, ref_day = map(int, self.ref_date_str.split('-'))
        ref_date = datetime(2000, ref_month, ref_day)

        for mod, cfg in self.modalities.items():
            if not mod.endswith("_TS"):
                continue

            if not cfg.get("dates_txt"):
                raise ValueError(f"{mod}: dates_file is required for every temporal input")

            if cfg.get("dates_txt"):
                with open(cfg["dates_txt"], 'r') as f:
                    date_strs = [line.strip() for line in f if line.strip()]
                if not date_strs:
                    raise ValueError(f"[✗] 'dates_txt' file for '{mod}' is empty.")

                dates = [datetime.strptime(d, "%Y%m%d") for d in date_strs]
                date_diffs = [(d - datetime(d.year, ref_date.month, ref_date.day)).days for d in dates]
                diff_dates[mod] = {
                    "dates": pd.Series(dates),
                    "diff_dates": np.array(date_diffs)
                }

        return diff_dates

    def _load_patch(
        self,
        reader: DatasetReader,
        bounds: Tuple[float, float, float, float],
        cfg: Dict[str, Any],
        patch_size: int,
        mod_name: Optional[str] = None
    ) -> Tuple[np.ndarray, Any]:
        """Load a raster patch using given bounds and config."""
        window = from_bounds(*bounds, transform=reader.transform)

        if mod_name and mod_name.endswith("_TS") and mod_name in self.diff_dates:
            num_dates = len(self.diff_dates[mod_name]['dates'])
            indexes = temporal_band_indexes(reader.count, num_dates, cfg['input_band_order'])
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

    def _normalize_patch(self, patch: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        """Apply the training run's band order and normalization."""
        norm_cfg = cfg.get('normalization', {})
        return prepare_model_input(
            patch,
            loaded_band_order=cfg.get("loaded_band_order", cfg.get("channels", [])),
            input_band_order=cfg.get("input_band_order", cfg.get("channels", [])),
            normalization_type=norm_cfg.get("type", "without"),
            mean=norm_cfg.get("mean", []),
            std=norm_cfg.get("std", []),
            scale=norm_cfg.get('scale', 1.0), offset=norm_cfg.get('offset', 0.0),
        )

    def _process_time_series_patch(
        self,
        mod_name: str,
        patch: np.ndarray,
        window: Any,
        cfg: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process and optionally filter a time series raster patch."""
        patch = reshape_sentinel(patch, chunk_size=len(cfg['channels']))
        date_info = self.diff_dates[mod_name]
        dates = date_info['dates']
        diff_dates = date_info['diff_dates']

        mask = None
        if mod_name == 'SENTINEL2_TS' and self.mask_reader:
            from rasterio.windows import bounds as window_bounds
            bounds = window_bounds(window, self.readers[mod_name].transform)
            mask_window = from_bounds(*bounds, transform=self.mask_reader.transform)
            mask_h = max(1, round(mask_window.height))
            mask_w = max(1, round(mask_window.width))
            count = 2 * len(dates)
            if self.mask_reader.count != count:
                raise ValueError('Cloud mask band count does not match acquisition dates')
            mask = self.mask_reader.read(
                indexes=list(range(1, count + 1)), window=mask_window,
                out_shape=(count, mask_h, mask_w), resampling=Resampling.nearest,
                boundless=True, fill_value=100)
            mask = reshape_sentinel(mask, 2)
        return prepare_time_series(
            patch, dates, diff_dates, mask=mask,
            max_cloud=cfg.get('max_cloud', 1), max_snow=cfg.get('max_snow', 1),
            maximum_covered_fraction=cfg.get('maximum_covered_fraction', .05),
            period=cfg.get('temporal_average', False), ref_date=self.ref_date_str)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample containing multimodal raster patches and metadata."""
        row = self.df.iloc[idx]
        bounds = row.geometry.bounds
        tile_data: Dict[str, torch.Tensor] = {}

        for mod_name, cfg in self.modalities.items():
            reader = self.readers[mod_name]
            patch_size = self.patch_sizes[mod_name]
            patch, window = self._load_patch(reader, bounds, cfg, patch_size, mod_name)

            if mod_name.endswith("_TS"):
                patch, diff_dates = self._process_time_series_patch(mod_name, patch, window, cfg)

                temporal_cfg = dict(cfg, loaded_band_order=cfg['input_band_order'])
                patch = self._normalize_patch(patch, temporal_cfg)
                tile_data[mod_name] = torch.tensor(patch, dtype=torch.float32)
                tile_data[mod_name.replace('_TS', '_DATES')] = torch.tensor(
                    np.ascontiguousarray(diff_dates), dtype=torch.float32
                )
            else:
                if mod_name == "DEM_ELEV":
                    patch = reorder_bands(patch, cfg['loaded_band_order'], cfg['input_band_order'])
                    cfg = dict(cfg, loaded_band_order=cfg['input_band_order'])
                if mod_name == "DEM_ELEV" and cfg.get("derive_elevation", False):
                    derived = calc_elevation(patch.astype(np.float32))
                    patch = (
                        np.stack((patch[0], derived[0]), axis=0)
                        if cfg.get("stack_with_dsm", False)
                        else derived
                    )
                patch = self._normalize_patch(patch, cfg)
                tile_data[mod_name] = torch.tensor(np.ascontiguousarray(patch), dtype=torch.float32)


        tile_data['index'] = torch.tensor([idx], dtype=torch.long)

        for task in self.modalities_config["labels"]:
            ref_patch_size = max(self.patch_sizes.values())
            tile_data[task] = torch.zeros((1, ref_patch_size, ref_patch_size), dtype=torch.float32)

        return tile_data

    def __del__(self) -> None:
        """Close all raster readers."""
        for reader in getattr(self, "readers", {}).values():
            reader.close()
        if getattr(self, "mask_reader", None):
            self.mask_reader.close()

