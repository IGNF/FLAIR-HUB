import sys
import numpy as np
import rasterio
import datetime
import pandas as pd
import torch

from skimage import img_as_float
from typing import Dict, Tuple
from torch.utils.data import Dataset



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




class flair_dataset(Dataset):
    """
    A PyTorch Dataset for handling multimodal remote sensing data, including aerial imagery,
    Sentinel satellite data, SPOT satellite data and elevation data. Supports data normalization, temporal 
    aggregation, filtering based on cloud and snow cover, and optional augmentations.
    Args:
        config (Dict): Configuration dictionary containing model and modality-specific settings, such as
            input channels, normalization parameters, and temporal processing options.
        dict_paths (Dict): A dictionary mapping data modalities (e.g., "AERIAL_RGBI", "LABELS") 
            to their corresponding file paths or metadata. Includes:
            - "AERIAL_RGBI": List of aerial RGBI patch file paths.
            - "AERIAL-RLT_PAN": List of aerial RLT PAN patch file paths.
            - "DEM_ELEV": List of DEM elevation patch file paths.
            - "SPOT_RGBI": List of SPOT RGBI patch file paths.
            - "SENTINEL2_TS": List of Sentinel-2 time-series patch file paths.
            - "SENTINEL2_MSK-SC": List of Sentinel-2 mask file paths.
            - "SENTINEL1-ASC_TS": List of Sentinel-1 ascending pass patch file paths.
            - "SENTINEL1-DESC_TS": List of Sentinel-1 descending pass patch file paths.
            - "LABELS": List of label file paths.
            - "DATES_S2": Metadata for Sentinel-2 dates and differences.
            - "DATES_S1_ASC": Metadata for Sentinel-1 ascending dates and differences.
            - "DATES_S1_DESC": Metadata for Sentinel-1 descending dates and differences.
            - "MTD": Optional metadata for samples (if use_metadata=True).
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
            Retrieves a sample with all modalities and metadata (if available). Applies 
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
        # Configuration
        self.config = config

        # Data paths
        self.list_patch_aerial = np.array(dict_paths["AERIAL_RGBI"])
        self.list_patch_aerial_rlt = np.array(dict_paths["AERIAL-RLT_PAN"])
        self.list_patch_elev = np.array(dict_paths["DEM_ELEV"])
        self.list_patch_spot = np.array(dict_paths["SPOT_RGBI"])
        self.list_patch_sentinel2 = np.array(dict_paths["SENTINEL2_TS"])
        self.list_patch_sentinel2msk = np.array(dict_paths["SENTINEL2_MSK-SC"])
        self.list_patch_sentinel1asc = np.array(dict_paths["SENTINEL1-ASC_TS"])
        self.list_patch_sentinel1desc = np.array(dict_paths["SENTINEL1-DESC_TS"])
        self.list_patch_label = np.array(dict_paths["LABELS"])

        # Date information
        self.dict_dates_s2 = dict_paths["DATES_S2"]
        self.dict_dates_s1asc = dict_paths["DATES_S1_ASC"]
        self.dict_dates_s1desc = dict_paths["DATES_S1_DESC"]

        # Metadata usage
        self.use_metadata = config['modalities']['pre_processings']['use_metadata']
        if self.use_metadata:
            self.list_metadata = np.array(dict_paths["MTD"])

        # Augmentations
        self.use_augmentations = use_augmentations

        # Sentinel filtering and temporal averaging
        self.filter_sentinel2 = config['modalities']['pre_processings']['filter_sentinel2']
        self.mth_average_sentinel2 = config['modalities']['pre_processings']['temporal_average_sentinel2']
        self.mth_average_sentinel1 = config['modalities']['pre_processings']['temporal_average_sentinel1']
        self.ref_date = config['models']['multitemp_model']['ref_date']

        # Channel configurations
        self.channels_aerial = config['modalities']['inputs_channels']['aerial']
        self.channels_spot = config['modalities']['inputs_channels']['spot']
        self.channels_sentinel2 = config['modalities']['inputs_channels']['sentinel2']
        self.channels_sentinel1 = config['modalities']['inputs_channels']['sentinel1']

        # Normalization parameters
        self.norm_type = config['modalities']['normalization']['norm_type']
        
        self.aerial_means = config['modalities']['normalization']['aerial_means']
        self.aerial_stds = config['modalities']['normalization']['aerial_stds']
        self.aerial_rlt_means = config['modalities']['normalization']['aerial_means']
        self.aerial_rlt_stds = config['modalities']['normalization']['aerial_stds']
        self.spot_means = config['modalities']['normalization']['spot_means']
        self.spot_stds = config['modalities']['normalization']['spot_stds']
        self.elev_means = config['modalities']['normalization']['elev_means']
        self.elev_stds = config['modalities']['normalization']['elev_stds']     

        # Class information
        self.num_classes = len(config['classes'])



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
        return len(self.list_patch_label)
    
    

    def __getitem__(self, index):
  
        sample = {
            'AERIAL_RGBI'          : [],
            'AERIAL-RLT_PAN'       : [],
            'DEM_ELEV'             : [],
            'SPOT_RGBI'            : [],
            'SENTINEL2_TS'         : [],
            'SENTINEL2_TS_DATES'   : [],
            'SENTINEL1-ASC_TS'     : [],
            'SENTINEL1-ASC_DATES'  : [],            
            'SENTINEL1-DESC_TS'    : [],
            'SENTINEL1-DESC_DATES' : [], 
            'LABELS'               : [],
            'MTD'                  : [],
            'ID'                   : self.list_patch_label[index],
        }
        

        if self.list_patch_aerial.size > 0:
            sample['AERIAL_RGBI'] = self.read_patch(self.list_patch_aerial[index], channels=self.channels_aerial)
            sample['AERIAL_RGBI'] = norm(sample['AERIAL_RGBI'], norm_type=self.aerial_norm_type, means=self.aerial_means, stds=self.aerial_stds)
            
        if self.list_patch_aerial_rlt.size > 0:
            sample['AERIAL-RLT_PAN'] = self.read_patch(self.list_patch_aerial_rlt[index])
            sample['AERIAL-RLT_PAN'] = norm(sample['AERIAL-RLT_PAN'], norm_type=self.norm_type, means=self.aerial_rlt_means, stds=self.aerial_rlt_stds)
   
        if self.list_patch_elev.size > 0:
            elev_data = self.read_patch(self.list_patch_elev[index])
            if self.config['modalities']['pre_processings']['calc_elevation']:
                sample['DEM_ELEV'] = self.calc_elevation(elev_data)
            else:
                sample['DEM_ELEV'] = elev_data
            sample['DEM_ELEV'] = norm(sample['DEM_ELEV'], norm_type=self.norm_type, means=self.elev_means, stds=self.elev_stds)
        
        if self.list_patch_spot.size > 0:
            sample['SPOT_RGBI'] = self.read_patch(self.list_patch_spot[index], channels=self.channels_spot)
            sample['SPOT_RGBI'] = norm(sample['SPOT_RGBI'], norm_type=self.norm_type, means=self.spot_means, stds=self.spot_stds)
        
        if self.list_patch_sentinel2.size > 0:
            sentinel2_data = self.read_patch(self.list_patch_sentinel2[index])
            sentinel2_data = self.reshape_sentinel(sentinel2_data, chunk_size=10)[:, [x - 1 for x in self.channels_sentinel2], :, :]
            sentinel2_dates_dict = self.dict_dates_s2['_'.join([sample['ID'].split('/')[-3].split('_')[0], sample['ID'].split('/')[-2]])]
            sentinel2_dates = sentinel2_dates_dict['dates']
            sentinel2_dates_diff = sentinel2_dates_dict['diff_dates'] 
            
            if self.filter_sentinel2 : 
                sentinel2msk_data = self.read_patch(self.list_patch_sentinel2msk[index]) 
                sentinel2msk_data = self.reshape_sentinel(sentinel2msk_data, chunk_size=2)  
                idx_valid = self.filter_time_series(sentinel2msk_data, 
                                                    max_cloud_value=self.config['modalities']['pre_processings']['filter_sentinel2_max_cloud'], 
                                                    max_snow_value=self.config['modalities']['pre_processings']['filter_sentinel2_max_snow'], 
                                                    max_fraction_covered=self.config['modalities']['pre_processings']['filter_sentinel2_max_frac_cover']
                            )
                sentinel2_data = sentinel2_data[np.where(idx_valid)[0]]
                sentinel2_dates = sentinel2_dates[np.where(idx_valid)[0]]
                sentinel2_dates_diff = sentinel2_dates_diff[np.where(idx_valid)[0]] 
            if self.mth_average_sentinel2:
                sentinel2_data, sentinel2_dates_diff = self.temporal_average(sentinel2_data, 
                                                                             sentinel2_dates, 
                                                                             period = self.mth_average_sentinel2,
                                                                             ref_date = self.ref_date 
                                                       )
            sample['SENTINEL2_TS'] = sentinel2_data
            sample['SENTINEL2_TS_DATES'] = sentinel2_dates_diff                  
            
        if self.list_patch_sentinel1asc.size > 0:
            sentinel1asc_data = self.read_patch(self.list_patch_sentinel1asc[index])
            sentinel1asc_data = self.reshape_sentinel(sentinel1asc_data, chunk_size=2)[:, [x - 1 for x in self.channels_sentinel1], :, :]
            sentinel1asc_dates_dict = self.dict_dates_s1asc['_'.join([sample['ID'].split('/')[-3].split('_')[0], sample['ID'].split('/')[-2]])]
            sentinel1asc_dates = sentinel1asc_dates_dict['dates']
            sentinel1asc_dates_diff = sentinel1asc_dates_dict['diff_dates'] 
            if self.mth_average_sentinel1:
                sentinel1asc_data, sentinel1asc_dates_diff = self.temporal_average(sentinel1asc_data, 
                                                                                   sentinel1asc_dates, 
                                                                                   period = self.mth_average_sentinel1,
                                                                                   ref_date = self.ref_date 
                                                             )   
            sample['SENTINEL1-ASC_TS'] = sentinel1asc_data
            sample['SENTINEL1-ASC_DATES'] = sentinel1asc_dates_diff                 
            
        if self.list_patch_sentinel1desc.size > 0:
            sentinel1desc_data = self.read_patch(self.list_patch_sentinel1desc[index])
            sentinel1desc_data = self.reshape_sentinel(sentinel1desc_data, chunk_size=2)[:, [x - 1 for x in self.channels_sentinel1], :, :]
            sentinel1desc_dates_dict = self.dict_dates_s1desc['_'.join([sample['ID'].split('/')[-3].split('_')[0], sample['ID'].split('/')[-2]])]
            sentinel1desc_dates = sentinel1desc_dates_dict['dates']
            sentinel1desc_dates_diff = sentinel1desc_dates_dict['diff_dates'] 
            if self.mth_average_sentinel1:
                sentinel1desc_data, sentinel1desc_dates_diff = self.temporal_average(sentinel1desc_data, 
                                                                                     sentinel1desc_dates, 
                                                                                     period = self.mth_average_sentinel1,
                                                                                     ref_date = self.ref_date 
                                                               )
            sample['SENTINEL1-DESC_TS'] = sentinel1desc_data
            sample['SENTINEL1-DESC_DATES'] = sentinel1desc_dates_diff  

        if self.list_patch_label.size > 0:
            label_data = self.read_patch(self.list_patch_label[index], channels=[1])
            label_data = label_data-1
            sample['LABELS'] = self.reshape_label_ohe(label_data, self.num_classes)
        
        if self.use_augmentations is not None:
            for key, value in sample.items():
                if len(value) == 0:
                    continue
                if len(value.shape) == 3:  # 3D 
                    sample[key] = value.swapaxes(0, 2).swapaxes(0, 1)
                elif len(value.shape) == 4:  # 4D
                    sample[key] = value.swapaxes(1, 3).swapaxes(1, 2)

            transformed_sample = self.use_augmentations(**sample)

            for key, value in transformed_sample.items():
                if len(value.shape) == 3:  
                    sample[key] = value.swapaxes(0, 2).swapaxes(1, 2).copy()
                elif len(value.shape) == 4: 
                    sample[key] = value.swapaxes(1, 2).swapaxes(2, 3).copy()            

        if self.use_metadata:
            sample['MTD'] = self.list_metadata[index]
        
        sample = {
            key: torch.as_tensor(value, dtype=torch.float) if len(value) > 0 and key != 'ID' else value 
            for key, value in sample.items()
        }
        
        return sample
