import numpy as np
import pandas as pd
import datetime

from typing import Tuple

def reshape_sentinel(arr: np.ndarray, chunk_size: int = 10) -> np.ndarray:
    first_dim_size = arr.shape[0] // chunk_size
    return arr.reshape((first_dim_size, chunk_size, *arr.shape[1:]))

def filter_time_series(
    data_array: np.ndarray, 
    max_cloud_value: int = 1, 
    max_snow_value: int = 1, 
    max_fraction_covered: float = 0.05
) -> np.ndarray:
    select = (data_array[:, 1, :, :] <= max_cloud_value) & (data_array[:, 0, :, :] <= max_snow_value)
    num_pix = data_array.shape[2] * data_array.shape[3]
    threshold = (1 - max_fraction_covered) * num_pix
    selected_idx = np.sum(select, axis=(1, 2)) >= threshold
    if not np.any(selected_idx):
        select = data_array[:, 0, :, :] <= max_snow_value
        selected_idx = np.sum(select, axis=(1, 2)) >= threshold
    return selected_idx

def _compute_monthly_average(
    data: np.ndarray, 
    df_dates: pd.DataFrame, 
    ref_datetime: datetime.datetime
) -> Tuple[np.ndarray, np.ndarray]:
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
            result.append(last_valid_month_data if last_valid_month_data is not None else np.zeros_like(data[0]))
            month_differences.append(month_differences[-1] if month_differences else 0)
    return np.array(result), np.array(month_differences)

def _compute_semi_monthly_average(
    data: np.ndarray, 
    df_dates: pd.DataFrame, 
    ref_datetime: datetime.datetime
) -> Tuple[np.ndarray, np.ndarray]:
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
                end_date = datetime.datetime(ref_datetime.year, month + 1, 1) - datetime.timedelta(days=1) if month < 12 else datetime.datetime(ref_datetime.year + 1, 1, 1) - datetime.timedelta(days=1)
                period_middle = datetime.datetime(ref_datetime.year, month, 23)
            indices = df_dates[(df_dates['dates'] >= start_date) & (df_dates['dates'] <= end_date)].index
            if len(indices) > 0:
                period_data = data[indices]
                semi_monthly_data.append(np.mean(period_data, axis=0))
                last_valid_period_data = np.mean(period_data, axis=0)
                period_diff = (period_middle - ref_datetime).days
                period_differences.append(period_diff)
            else:
                semi_monthly_data.append(last_valid_period_data if last_valid_period_data is not None else np.zeros_like(data[0]))
                period_differences.append(period_differences[-1] if period_differences else 0)
    return np.array(semi_monthly_data), np.array(period_differences)

def temporal_average(
    data: np.ndarray, 
    dates: pd.Series, 
    period: str = "monthly", 
    ref_date: str = "01-01"
) -> Tuple[np.ndarray, np.ndarray]:
    ref_month, ref_day = map(int, ref_date.split('-'))
    ref_year = dates[0].year
    ref_datetime = datetime.datetime(ref_year, ref_month, ref_day)
    df_dates = pd.DataFrame({'dates': dates})
    df_dates['month'] = df_dates['dates'].dt.month
    df_dates['day'] = df_dates['dates'].dt.day
    if period == "monthly":
        return _compute_monthly_average(data, df_dates, ref_datetime)
    elif period == "semi-monthly":
        return _compute_semi_monthly_average(data, df_dates, ref_datetime)
    else:
        raise ValueError("Period must be either 'monthly' or 'semi-monthly'.")
