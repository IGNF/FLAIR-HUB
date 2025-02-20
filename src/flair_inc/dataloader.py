import sys
import numpy as np
import rasterio
import datetime
import pandas as pd
import torch

from typing import Dict, Tuple
from torch.utils.data import Dataset

from .utils.utils_data import norm



class flair_dataset(Dataset):
    """
    A PyTorch Dataset for handling multimodal remote sensing data, including aerial imagery,
    Sentinel satellite data, SPOT satellite data and elevation data. Supports data normalization, temporal 
    aggregation, filtering based on cloud and snow cover, and optional augmentations.
    Args:
        config (Dict): Configuration dictionary containing model and modality-specific settings, such as
            input channels, normalization parameters, and temporal processing options.
        dict_paths (Dict): A dictionary mapping data modalities (e.g., "AERIAL_RGBI", "LABELS") 
            to their corresponding file paths or metadata. 
        use_augmentations (callable, optional): A callable function or transformation pipeline to 
            apply augmentations to the samples. Defaults to None.
    Methods:
        read_patch(raster_file: str, channels: list = None) -> np.ndarray:
            Reads a raster patch and extracts specified channels.
        reshape_sentinel(arr: np.ndarray, chunk_size: int = 10) -> np.ndarray:
            Reshapes Sentinel time-series data into temporal chunks.
        reshape_label_ohe(arr: np.ndarray, num_classes: int) -> np.ndarray:
            Converts label data into one-hot encoding.
        calc_elevation(arr: np.ndarray) -> np.ndarray:
            Calculates elevation differences from DEM elevation data.
        filter_time_series(data_array: np.ndarray, max_cloud_value: int = 1, 
                           max_snow_value: int = 1, max_fraction_covered: float = 0.05) -> np.ndarray:
            Filters Sentinel-2 data based on cloud/snow coverage thresholds.
        temporal_average(data: np.ndarray, dates: pd.Series, period: str = "monthly", 
                         ref_date: str = "01-01") -> Tuple[np.ndarray, np.ndarray]:
            Computes temporal averages for monthly or semi-monthly intervals and differences 
            from a reference date.
        __len__() -> int:
            Returns the number of samples in the dataset.
        __getitem__(index: int) -> Dict:
            Retrieves batch_elements. Applies 
            normalization, temporal aggregation, and augmentations (if specified).
    """


    def __init__(
        self,
        config: Dict,
        dict_paths: Dict,
        use_augmentations: bool = None,
    ) -> None:
        """
        Initialize the flair_dataset instance.

        Parameters:
        - config (Dict): Configuration dictionary containing settings for the dataset.
        - dict_paths (Dict): Dictionary with paths to various data patches and metadata.
        - use_augmentations (bool, optional): Flag to indicate if augmentations should be applied. Defaults to None.
        """

        self.config = config

        # Initialize list_patch dynamically based on enabled modalities in the config
        self.list_patch = {}
        enabled_modalities = self.config.get("modalities", {}).get("inputs", {})
        for modality, is_enabled in enabled_modalities.items():
            if is_enabled: 
                if modality in dict_paths:
                    self.list_patch[modality] = np.array(dict_paths[modality])
                    if modality == 'SENTINEL2_TS':
                        self.list_patch["SENTINEL2_MSK-SC"] = np.array(dict_paths["SENTINEL2_MSK-SC"])

        # Supervision
        self.tasks = {}
        for task in config['labels']:
            task_dict = {
                'data_paths': np.array(dict_paths[task]),
                'num_classes': len(config['labels_configs'][task]['value_name'])
            }
            if 'label_channel_nomenclature' in config['labels_configs'][task]:
                task_dict['channels'] = [config['labels_configs'][task]['label_channel_nomenclature']]
            else:
                task_dict['channels'] = [1]
            self.tasks[task] = task_dict

        # Date information (only for Sentinel modalities)
        self.dict_dates = {}
        if 'SENTINEL2_TS' in enabled_modalities:
            self.dict_dates['SENTINEL2_TS'] = dict_paths.get("DATES_S2", {})
        if 'SENTINEL1-ASC_TS' in enabled_modalities:
            self.dict_dates['SENTINEL1-ASC_TS'] = dict_paths.get("DATES_S1_ASC", {})
        if 'SENTINEL1-DESC_TS' in enabled_modalities:
            self.dict_dates['SENTINEL1-DESC_TS'] = dict_paths.get("DATES_S1_DESC", {})

        # Sentinel filtering and temporal averaging (only if relevant Sentinel modalities are enabled)
        self.filter_sentinel2 = config['modalities']['pre_processings']['filter_sentinel2']
        self.mth_average_sentinel2 = config['modalities']['pre_processings']['temporal_average_sentinel2']
        self.mth_average_sentinel1 = config['modalities']['pre_processings']['temporal_average_sentinel1']
        self.ref_date = config['models']['multitemp_model']['ref_date']

        # Channel Configurations
        self.channels = {
            modality: config['modalities']['inputs_channels'].get(modality, [])
            for modality in enabled_modalities if enabled_modalities.get(modality)
        }

        # Normalization parameters
        self.norm_type = config['modalities']['normalization']['norm_type']

        self.normalization = {
            modality: {
                'mean': config['modalities']['normalization'].get(f'{modality}_means', []),
                'std': config['modalities']['normalization'].get(f'{modality}_stds', [])
            }
            for modality in enabled_modalities if enabled_modalities.get(modality)
        }

        # Augmentations
        self.use_augmentations = use_augmentations



    def read_patch(self, raster_file: str, channels: list = None) -> np.ndarray:
        """
        Reads patch data from a raster file.
        Parameters:
        - raster_file (str): Path to the raster file.
        - channels (list, optional): List of channel indices to read. If None, reads all channels.
        Returns:
        - np.ndarray: The extracted patch data.
        """
        with rasterio.open(raster_file) as src_img:
            array = src_img.read(channels) if channels else src_img.read()
        return array



    def reshape_sentinel(self, arr: np.ndarray, chunk_size: int = 10) -> np.ndarray:
        """
        Reshapes a temporally stacked Sentinel array into chunks.
        Parameters:
        - arr (np.ndarray): Input array with temporal data.
        - chunk_size (int, optional): Number of time steps per chunk. Defaults to 10.
        Returns:
        - np.ndarray: Reshaped array with shape (n_chunks, chunk_size, height, width).
        """
        first_dim_size = arr.shape[0] // chunk_size
        return arr.reshape((first_dim_size, chunk_size, *arr.shape[1:]))



    def reshape_label_ohe(self, arr: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Converts a label array into one-hot-encoded format.
        Parameters:
        - arr (np.ndarray): Input label array.
        - num_classes (int): Total number of classes.
        Returns:
        - np.ndarray: One-hot-encoded label array with shape (num_classes, ...).
        """
        if arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return np.stack([arr == i for i in range(num_classes)], axis=0)



    def calc_elevation(self, arr: np.ndarray) -> np.ndarray:
        """
        Calculates the elevation difference between two input channels.
        Parameters:
        - arr (np.ndarray): Input array where the first channel is elevation and the second is baseline.
        Returns:
        - np.ndarray: Array containing the elevation difference with shape (1, height, width).
        """
        elev = arr[0] - arr[1]
        return elev[np.newaxis, :, :]



    def filter_time_series(self, data_array: np.ndarray, max_cloud_value: int = 1, 
                        max_snow_value: int = 1, max_fraction_covered: float = 0.05) -> np.ndarray:
        """
        Filters time-series data based on cloud and snow coverage thresholds.
        Parameters:
        - data_array (np.ndarray): Array with shape (days, channels, x, y).
        - max_cloud_value (int, optional): Maximum allowed cloud value. Defaults to 1.
        - max_snow_value (int, optional): Maximum allowed snow value. Defaults to 1.
        - max_fraction_covered (float, optional): Maximum fraction of covered area allowed. Defaults to 0.05.
        Returns:
        - np.ndarray: Boolean array indicating selected days.
        """
        select = (data_array[:, 1, :, :] <= max_cloud_value) & (data_array[:, 0, :, :] <= max_snow_value)
        num_pix = data_array.shape[2] * data_array.shape[3]
        threshold = (1 - max_fraction_covered) * num_pix

        selected_idx = np.sum(select, axis=(1, 2)) >= threshold

        if not np.any(selected_idx):
            select = data_array[:, 0, :, :] <= max_snow_value
            selected_idx = np.sum(select, axis=(1, 2)) >= threshold

        return selected_idx


    def _compute_monthly_average(self, data: np.ndarray, df_dates: pd.DataFrame, 
                                ref_datetime: datetime.datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes monthly averages and differences from a reference date.
        Parameters:
        - data (np.ndarray): Input data array with shape (n_samples, n_features, ...).
        - df_dates (pd.DataFrame): DataFrame containing 'dates', 'month', and 'day' columns.
        - ref_datetime (datetime.datetime): Reference date for calculating differences.
        Returns:
        - Tuple[np.ndarray, np.ndarray]:
            - np.ndarray: Monthly averages with shape (12, ...).
            - np.ndarray: Day differences relative to the reference date (12,).
        """
        months = np.arange(1, 13)
        result = []
        month_differences = []
        last_valid_month_data = None

        for month in months:
            indices = df_dates[df_dates['month'] == month].index
            if len(indices) > 0:
                month_data = data[indices]
                result.append(np.mean(month_data, axis=0))
                last_valid_month_data = np.mean(month_data, axis=0)
                middle_of_month = datetime.datetime(ref_datetime.year, month, 15)
                month_diff = (middle_of_month - ref_datetime).days
                month_differences.append(month_diff)
            else:
                if last_valid_month_data is not None:
                    result.append(last_valid_month_data)
                else:
                    result.append(np.zeros_like(data[0]))
                month_differences.append(month_differences[-1] if month_differences else 0)

        return np.array(result), np.array(month_differences)

 
    def _compute_semi_monthly_average(self, data: np.ndarray, df_dates: pd.DataFrame, 
                                    ref_datetime: datetime.datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes semi-monthly averages and differences from a reference date.
        Parameters:
        - data (np.ndarray): Input data array with shape (n_samples, n_features, ...).
        - df_dates (pd.DataFrame): DataFrame containing 'dates', 'month', and 'day' columns.
        - ref_datetime (datetime.datetime): Reference date for calculating differences.
        Returns:
        - Tuple[np.ndarray, np.ndarray]:
            - np.ndarray: Semi-monthly averages with shape (24, ...).
            - np.ndarray: Day differences relative to the reference date (24,).
        """
        semi_monthly_data = []
        period_differences = []
        last_valid_period_data = None

        for month in np.arange(1, 13):
            for period_id in ['first_half', 'second_half']:
                if period_id == 'first_half':
                    start_date = datetime.datetime(ref_datetime.year, month, 1)
                    end_date = datetime.datetime(ref_datetime.year, month, 15)
                    period_middle = datetime.datetime(ref_datetime.year, month, 8)
                else:
                    start_date = datetime.datetime(ref_datetime.year, month, 16)
                    if month == 12:
                        end_date = datetime.datetime(ref_datetime.year + 1, 1, 1) - datetime.timedelta(days=1)
                    else:
                        end_date = datetime.datetime(ref_datetime.year, month + 1, 1) - datetime.timedelta(days=1)
                    period_middle = datetime.datetime(ref_datetime.year, month, 23)

                indices = df_dates[(df_dates['dates'] >= start_date) & (df_dates['dates'] <= end_date)].index
                if len(indices) > 0:
                    period_data = data[indices]
                    semi_monthly_data.append(np.mean(period_data, axis=0))
                    last_valid_period_data = np.mean(period_data, axis=0)
                    period_diff = (period_middle - ref_datetime).days
                    period_differences.append(period_diff)
                else:
                    if last_valid_period_data is not None:
                        semi_monthly_data.append(last_valid_period_data)
                    else:
                        semi_monthly_data.append(np.zeros_like(data[0]))
                    period_differences.append(period_differences[-1] if period_differences else 0)

        return np.array(semi_monthly_data), np.array(period_differences)


    def temporal_average(self, data: np.ndarray, dates: pd.Series, period: str = "monthly", 
                        ref_date: str = "01-01") -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes temporal averages and differences relative to a reference date.
        Parameters:
        - data (np.ndarray): Input data array with shape (n_samples, n_features, ...).
        - dates (pd.Series): Series of datetime objects corresponding to each sample.
        - period (str, optional): Averaging period: "monthly" or "semi-monthly". Defaults to "monthly".
        - ref_date (str, optional): Reference date in 'MM-DD' format. Defaults to "01-01".
        Returns:
        - Tuple[np.ndarray, np.ndarray]:
            - np.ndarray: Averages for each period (shape depends on `period`).
            - np.ndarray: Day differences relative to the reference date.
        """
        ref_month, ref_day = map(int, ref_date.split('-'))
        ref_year = dates[0].year
        ref_datetime = datetime.datetime(ref_year, ref_month, ref_day)

        df_dates = pd.DataFrame({'dates': dates})
        df_dates['month'] = df_dates['dates'].dt.month
        df_dates['day'] = df_dates['dates'].dt.day

        if period == "monthly":
            return self._compute_monthly_average(data, df_dates, ref_datetime)
        elif period == "semi-monthly":
            return self._compute_semi_monthly_average(data, df_dates, ref_datetime)
        else:
            raise ValueError("Period must be either 'monthly' or 'semi-monthly'.")



    def __len__(self):
        # Find the first non-empty supervision list of paths
        for task, task_info in self.tasks.items():
            if len(task_info['data_paths']) > 0:
                return len(task_info['data_paths'])
        # Fallback in case all lists are empty
        return 0
    
    

    def __getitem__(self, index):

        batch_elements = {}

        # Process supervision tasks
        for task, task_info in self.tasks.items():
            if len(task_info['data_paths']) > 0:
                batch_elements[f'ID_{task}'] = task_info['data_paths'][index]
                area_elem = task_info['data_paths'][index].split('/')[-1].split('_')
                area_elem = '_'.join([area_elem[0], area_elem[-2]])


        # Process AERIAL_RGBI data
        KEY='AERIAL_RGBI'
        if self.list_patch.get(KEY, np.array([])).size > 0:
            batch_elements[KEY] = self.read_patch(self.list_patch[KEY][index], channels=self.channels[KEY])
            batch_elements[KEY] = norm(batch_elements[KEY], norm_type=self.norm_type, means=self.normalization[KEY]['mean'], stds=self.normalization[KEY]['std'])


        # Process AERIAL-RLT_PAN data
        KEY='AERIAL-RLT_PAN'
        if self.list_patch.get(KEY, np.array([])).size > 0:
            batch_elements[KEY] = self.read_patch(self.list_patch[KEY][index])
            batch_elements[KEY] = norm(batch_elements[KEY], norm_type=self.norm_type, means=self.normalization[KEY]['mean'], stds=self.normalization[KEY]['std'])


        # Process DEM_ELEV data
        KEY='DEM_ELEV'
        if self.list_patch.get(KEY, np.array([])).size > 0:
            zdata = self.read_patch(self.list_patch[KEY][index])
            if self.config['modalities']['pre_processings']['calc_elevation']:
                elev_data = self.calc_elevation(zdata)
                if self.config['modalities']['pre_processings']['calc_elevation_stack_dsm']:
                    elev_data = np.stack((zdata[0, :, :], elev_data[0]), axis=0)
                batch_elements[KEY] = elev_data
            else:
                batch_elements[KEY] = zdata
            batch_elements[KEY] = norm(batch_elements[KEY], norm_type=self.norm_type, means=self.normalization[KEY]['mean'], stds=self.normalization[KEY]['std'])


        # Process SPOT_RGBI data
        KEY='SPOT_RGBI'
        if self.list_patch.get(KEY, np.array([])).size > 0:
            batch_elements[KEY] = self.read_patch(self.list_patch[KEY][index], channels=self.channels[KEY])
            batch_elements[KEY] = norm(batch_elements[KEY], norm_type=self.norm_type, means=self.normalization[KEY]['mean'], stds=self.normalization[KEY]['std'])


        # Process SENTINEL2 data
        KEY='SENTINEL2_TS'
        if self.list_patch.get(KEY, np.array([])).size > 0:
            sentinel2_data = self.read_patch(self.list_patch[KEY][index])
            sentinel2_data = self.reshape_sentinel(sentinel2_data, chunk_size=10)[:, [x - 1 for x in self.channels[KEY]], :, :]
            sentinel2_dates_dict = self.dict_dates[KEY][area_elem]
            sentinel2_dates = sentinel2_dates_dict['dates']
            sentinel2_dates_diff = sentinel2_dates_dict['diff_dates']

            if self.filter_sentinel2:
                sentinel2msk_data = self.read_patch(self.list_patch['SENTINEL2_MSK-SC'][index])
                sentinel2msk_data = self.reshape_sentinel(sentinel2msk_data, chunk_size=2)
                idx_valid = self.filter_time_series(sentinel2msk_data,
                                                    max_cloud_value=self.config['modalities']['pre_processings']['filter_sentinel2_max_cloud'],
                                                    max_snow_value=self.config['modalities']['pre_processings']['filter_sentinel2_max_snow'],
                                                    max_fraction_covered=self.config['modalities']['pre_processings']['filter_sentinel2_max_frac_cover'])
                sentinel2_data = sentinel2_data[np.where(idx_valid)[0]]
                sentinel2_dates = sentinel2_dates[np.where(idx_valid)[0]]
                sentinel2_dates_diff = sentinel2_dates_diff[np.where(idx_valid)[0]]

            if self.mth_average_sentinel2:
                sentinel2_data, sentinel2_dates_diff = self.temporal_average(sentinel2_data,
                                                                            sentinel2_dates,
                                                                            period=self.mth_average_sentinel2,
                                                                            ref_date=self.ref_date)
            batch_elements[KEY] = sentinel2_data
            batch_elements[KEY.replace('_TS', '_DATES')] = sentinel2_dates_diff



        # Process SENTINEL1 ASC data
        KEY='SENTINEL1-ASC_TS'
        if self.list_patch.get(KEY, np.array([])).size > 0:
            sentinel1asc_data = self.read_patch(self.list_patch[KEY][index])
            sentinel1asc_data = self.reshape_sentinel(sentinel1asc_data, chunk_size=2)[:, [x - 1 for x in self.channels[KEY]], :, :]
            sentinel1asc_dates_dict = self.dict_dates[KEY][area_elem]
            sentinel1asc_dates = sentinel1asc_dates_dict['dates']
            sentinel1asc_dates_diff = sentinel1asc_dates_dict['diff_dates']

            if self.mth_average_sentinel1:
                sentinel1asc_data, sentinel1asc_dates_diff = self.temporal_average(sentinel1asc_data,
                                                                                sentinel1asc_dates,
                                                                                period=self.mth_average_sentinel1,
                                                                                ref_date=self.ref_date)
            batch_elements[KEY] = sentinel1asc_data
            batch_elements[KEY.replace('_TS', '_DATES')] = sentinel1asc_dates_diff



        # Process SENTINEL1 DESC data
        KEY='SENTINEL1-DESC_TS'
        if self.list_patch.get(KEY, np.array([])).size > 0:
            sentinel1desc_data = self.read_patch(self.list_patch[KEY][index])
            sentinel1desc_data = self.reshape_sentinel(sentinel1desc_data, chunk_size=2)[:, [x - 1 for x in self.channels[KEY]], :, :]
            sentinel1desc_dates_dict = self.dict_dates[KEY][area_elem]
            sentinel1desc_dates = sentinel1desc_dates_dict['dates']
            sentinel1desc_dates_diff = sentinel1desc_dates_dict['diff_dates']

            if self.mth_average_sentinel1:
                sentinel1desc_data, sentinel1desc_dates_diff = self.temporal_average(sentinel1desc_data,
                                                                                sentinel1desc_dates,
                                                                                period=self.mth_average_sentinel1,
                                                                                ref_date=self.ref_date)

            batch_elements[KEY] = sentinel1desc_data
            batch_elements[KEY.replace('_TS', '_DATES')] = sentinel1desc_dates_diff



        # Process supervision tasks 
        for task, task_info in self.tasks.items():
            label_data = self.read_patch(task_info['data_paths'][index], channels=task_info['channels'])
            batch_elements[task] = self.reshape_label_ohe(label_data, task_info['num_classes'])



        # Apply augmentations 
        if self.use_augmentations is not None:
            for key, value in batch_elements.items():
                if len(value) == 0:
                    continue
                if len(value.shape) == 3:  # 3D
                    batch_elements[key] = value.swapaxes(0, 2).swapaxes(0, 1)
                elif len(value.shape) == 4:  # 4D
                    batch_elements[key] = value.swapaxes(1, 3).swapaxes(1, 2)

            transformed_sample = self.use_augmentations(**batch_elements)

            for key, value in transformed_sample.items():
                if len(value.shape) == 3:  
                    batch_elements[key] = value.swapaxes(0, 2).swapaxes(1, 2).copy()
                elif len(value.shape) == 4: 
                    batch_elements[key] = value.swapaxes(1, 2).swapaxes(2, 3).copy()

        

        batch_elements = {
            key: torch.as_tensor(value, dtype=torch.float) if isinstance(value, (list, np.ndarray)) and 'ID_' not in key else value
            for key, value in batch_elements.items()
        }

        return batch_elements



