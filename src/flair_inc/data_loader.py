import numpy as np
import rasterio
import datetime
import pandas as pd
import torch

from skimage import img_as_float
from typing import Dict, Tuple
from torch.utils.data import Dataset


def norm(in_img : np.ndarray, 
         norm_type : str = None, 
         means : list = [], 
         stds: list = [],
         ):
     
    if norm_type not in ['scaling','custom','without']:
            print("Normalization argument should be 'scaling', 'custom' or 'without'.")
            raise SystemExit()
    if norm_type: 
        if norm_type == 'custom':
            if len(means) != len(stds):
                print("If custom, provided normalization means and stds should be of same lenght.")
                raise SystemExit()                
            else:
                in_img = in_img.astype(np.float64)
                for i in range(in_img.shape[0]):
                    in_img[i] -= means[i]
                    in_img[i] /= stds[i]
        elif norm_type == 'scaling':
            in_img = img_as_float(in_img)
    return in_img



class flair_dataset(Dataset):

    def __init__(self,
                 config : Dict, 
                 dict_paths : Dict,
                 use_augmentations : bool = None,
                 ) -> None:
        
        self.config = config 
        self.list_patch_aerial = np.array(dict_paths["AERIAL_RGBI"])
        self.list_patch_aerial_rlt = np.array(dict_paths["AERIAL-RLT_PAN"])
        self.list_patch_elev = np.array(dict_paths["DEM_ELEV"])       
        self.list_patch_spot = np.array(dict_paths["SPOT_RGBI"])
        self.list_patch_sentinel2 = np.array(dict_paths["SENTINEL2_TS"])
        self.list_patch_sentinel2msk = np.array(dict_paths["SENTINEL2_MSK-SC"])
        self.list_patch_sentinel1asc = np.array(dict_paths["SENTINEL1-ASC_TS"])
        self.list_patch_sentinel1desc = np.array(dict_paths["SENTINEL1-DESC_TS"])
        self.list_patch_label = np.array(dict_paths["LABELS"])
        
        self.dict_dates_s2 = dict_paths["DATES_S2"]
        self.dict_dates_s1asc = dict_paths["DATES_S1_ASC"]
        self.dict_dates_s1desc = dict_paths["DATES_S1_DESC"]

        self.use_metadata = config['modalities']['use_metadata']
        if self.use_metadata == True:
            self.list_metadata = np.array(dict_paths["MTD"])
            
        self.use_augmentations = use_augmentations
        
        self.filter_sentinel2 = config['modalities']['filter_sentinel2']
        self.mth_average_sentinel2 = config['modalities']['temporal_average_sentinel2']
        self.mth_average_sentinel1 = config['modalities']['temporal_average_sentinel1']
        self.ref_date = config['models']['multitemp_model']['ref_date']
        
        self.channels_aerial = config['modalities']['inputs_channels']['aerial']
        self.channels_spot = config['modalities']['inputs_channels']['spot']        
        self.channels_sentinel2 = config['modalities']['inputs_channels']['sentinel2']        
        self.channels_sentinel1 = config['modalities']['inputs_channels']['sentinel1']        

        self.aerial_norm_type= config['modalities']['normalization']['norm_type']
        self.aerial_means = config['modalities']['normalization']['aerial_means']
        self.aerial_stds = config['modalities']['normalization']['aerial_stds']
        
        self.num_classes = len(config['classes'])


    def read_patch(self, raster_file: str, channels: list = None) -> np.ndarray:
        """
        Read patch file.
        Args:
            raster_file (str): Path to the raster file.
            channels (list, optional): List of channels to read. If not provided, all channels will be read.
        Returns:
            np.ndarray: The patch data.
        """    
        with rasterio.open(raster_file) as src_img:
            if channels is not None:
                array = src_img.read(channels)
            else:
                array = src_img.read()  
        return array

    def reshape_sentinel(self, arr: np.ndarray, chunk_size: int = 10) -> np.ndarray:
        """
        Reshape the temporally stacked sentinel files. Chunk size varies with number of channels of files. 
        """    
        first_dim_size = arr.shape[0] // chunk_size
        arr = arr.reshape((first_dim_size, chunk_size, arr.shape[1], arr.shape[2]))
        return arr

    def reshape_label_ohe(self, arr: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Reshape labels files to one-hot-encoding for loss computation. 
        """      
        arr = np.stack([arr == i for i in range(num_classes)], axis=0)
        return arr

    def calc_elevation(self, arr: np.ndarray) -> np.ndarray:
        elev = arr[0] - arr[1]  
        arr = elev[np.newaxis, :, :]
        return arr

    def filter_time_series(self, 
                           data_array: np.ndarray, 
                           max_cloud_value: int = 1, 
                           max_snow_value: int = 1, 
                           max_fraction_covered: float = 0.05) -> np.ndarray:
        """
        Filters out cloudy and snowy days given a cloud and snow model stored in channel numbers 10 and 11.
        A maximum of 'max_fraction_covered' of the parcel surface has a cloud value larger than 'max_cloud_value'
        and a snow value larger than 'max_snow_value'.
        Parameters:
        - data_array: masks numpy array of shape (days, channels, x_coord, y_coord)
        - max_cloud_value: maximum allowed cloud value
        - max_snow_value: maximum allowed snow value
        - max_fraction_covered: maximum fraction of the surface that can be covered by clouds/snow
        Returns:
        - selected_idx: boolean array indicating which days were selected
        """
        
        select = (data_array[:, 1, :, :] <= max_cloud_value) & (data_array[:, 0, :, :] <= max_snow_value)
        num_pix = data_array.shape[2] * data_array.shape[3]
        threshold = (1 - max_fraction_covered) * num_pix

        selected_idx = np.sum(select, axis=(1, 2)) >= threshold

        if not np.any(selected_idx):
            #print("All days were filtered out; retrying without cloud filter.")
            select = data_array[:, 0, :, :] <= max_snow_value
            selected_idx = np.sum(select, axis=(1, 2)) >= threshold

        return selected_idx

    def temporal_average(self, data: np.ndarray, dates: pd.Series, period: str = "monthly", ref_date: str = "01-01") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute temporal averages for monthly or semi-monthly periods and calculate the difference from a reference date.
        Parameters:
        - data (np.ndarray): The data array to average, typically with shape (n_samples, n_features, ...).
        - dates (pd.Series): A pandas Series containing the datetime objects for each sample.
        - period (str, optional): The period for averaging, either "monthly" or "semi-monthly". Defaults to "monthly".
        - ref_date (str, optional): The reference date in 'MM-DD' format. The difference for each period is calculated relative to this date. Defaults to "01-01".
        Returns:
        - Tuple[np.ndarray, np.ndarray]:
            - np.ndarray: The temporal averages for each period, with shape (12, ...) for monthly or (24, ...) for semi-monthly.
            - np.ndarray: The calculated day differences for each period relative to the reference date, with shape (12,) or (24,).
        Raises:
        - ValueError: If the period is not "monthly" or "semi-monthly".
        """
        ref_month, ref_day = map(int, ref_date.split('-'))
        ref_year = dates[0].year  
        ref_datetime = datetime.datetime(ref_year, ref_month, ref_day)
        df_dates = pd.DataFrame({'dates': dates})
        df_dates['month'] = df_dates['dates'].dt.month
        df_dates['day'] = df_dates['dates'].dt.day

        if period == "monthly":
            months = np.arange(1, 13)
            result = []
            last_valid_month_data = None
            month_differences = []
            for month in months:
                indices = df_dates[df_dates['month'] == month].index
                if len(indices) > 0:  
                    month_data = data[indices]  
                    result.append(np.mean(month_data, axis=0))  
                    last_valid_month_data = np.mean(month_data, axis=0)  
                    middle_of_month = datetime.datetime(ref_year, month, 15)
                    month_diff = (middle_of_month - ref_datetime).days  
                    month_differences.append(month_diff)
                else:
                    if last_valid_month_data is not None:
                        result.append(last_valid_month_data)
                    else:
                        result.append(np.zeros_like(data[0]))  
                    month_differences.append(month_differences[-1] if month_differences else 0)  
            result_array = np.array(result)
            return result_array, np.array(month_differences)

        elif period == "semi-monthly":
            semi_monthly_data = []
            last_valid_period_data = None
            result = []
            period_differences = []
            for month in np.arange(1, 13):
                for period_id in ['first_half', 'second_half']:
                    if period_id == 'first_half':
                        start_date = pd.Timestamp(datetime.datetime(ref_year, month, 1))
                        end_date = pd.Timestamp(datetime.datetime(ref_year, month, 15))
                    elif period_id == 'second_half':
                        start_date = pd.Timestamp(datetime.datetime(ref_year, month, 16))
                        if month == 12:
                            end_date = pd.Timestamp(datetime.datetime(ref_year + 1, 1, 1)) - pd.Timedelta(days=1)
                        else:
                            end_date = pd.Timestamp(datetime.datetime(ref_year, month + 1, 1)) - pd.Timedelta(days=1)
                    indices = df_dates[(df_dates['dates'] >= start_date) & (df_dates['dates'] <= end_date)].index
                    if len(indices) > 0:  
                        period_data = data[indices]  
                        result.append(np.mean(period_data, axis=0))  
                        last_valid_period_data = np.mean(period_data, axis=0)  
                        if period_id == 'first_half':
                            period_middle = datetime.datetime(ref_year, month, 8)  
                        else:
                            period_middle = datetime.datetime(ref_year, month, 23)  
                        period_diff = (period_middle - ref_datetime).days  
                        period_differences.append(period_diff)
                    else:
                        if last_valid_period_data is not None:
                            result.append(last_valid_period_data)
                        else:
                            result.append(np.zeros_like(data[0]))  
                        period_differences.append(period_differences[-1] if period_differences else 0)  
            result_array = np.array(result)
            return result_array, np.array(period_differences)
        else:
            raise ValueError("Period must be either 'monthly' or 'semi-monthly'.")



    def __len__(self):
        return len(self.list_patch_aerial)
    
    

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
            ## Norm - TODO : ADD FOR OTHERS ? 
            sample['AERIAL_RGBI'] = norm(sample['AERIAL_RGBI'], norm_type=self.aerial_norm_type, means=self.aerial_means, stds=self.aerial_stds)
            
        if self.list_patch_aerial_rlt.size > 0:
            sample['AERIAL-RLT_PAN'] = self.read_patch(self.list_patch_aerial_rlt[index])
        
        if self.list_patch_elev.size > 0:
            elev_data = self.read_patch(self.list_patch_elev[index])
            sample['DEM_ELEV'] = self.calc_elevation(elev_data)
        
        if self.list_patch_spot.size > 0:
            sample['SPOT_RGBI'] = self.read_patch(self.list_patch_spot[index], channels=self.channels_spot)
        
        if self.list_patch_sentinel2.size > 0:
            sentinel2_data = self.read_patch(self.list_patch_sentinel2[index])
            sentinel2_data = self.reshape_sentinel(sentinel2_data, chunk_size=10)[:, [x - 1 for x in self.channels_sentinel2], :, :]
            sentinel2_dates_dict = self.dict_dates_s2['_'.join([sample['ID'].split('/')[-3].split('_')[0], sample['ID'].split('/')[-2]])]
            sentinel2_dates = sentinel2_dates_dict['dates']
            sentinel2_dates_diff = sentinel2_dates_dict['diff_dates'] 
            
            if self.filter_sentinel2 : 
                sentinel2msk_data = self.read_patch(self.list_patch_sentinel2msk[index]) 
                sentinel2msk_data = self.reshape_sentinel(sentinel2msk_data, chunk_size=2)  
                idx_valid = self.filter_time_series(sentinel2msk_data, max_cloud_value=1, max_snow_value=1, max_fraction_covered=0.05)
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