import numpy as np
import torch

from torch.nn import functional as F


def _calculate_padding_length(x: np.ndarray, l: int) -> int:
    """
    Calculate the required padding length based on the target length.
    Parameters:
    - x (np.ndarray): The input tensor to be padded.
    - l (int): The target length for the first dimension (axis 0).
    Returns:
    - int: The padding length for the first dimension (axis 0).
    """
    padlen = l - x.shape[0]
    return padlen


def _create_padding_array(x: np.ndarray, padlen: int) -> list:
    """
    Creates a padding array to be used with `F.pad`.
    Parameters:
    - x (np.ndarray): The input tensor to be padded.
    - padlen (int): The padding length for the first dimension (axis 0).
    Returns:
    - list: The padding array to use with `F.pad`.
    """
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return pad


def pad_tensor(x: np.ndarray, l: int, pad_value: int = 0) -> np.ndarray:
    """
    Pads the tensor `x` along the first dimension to the target length `l`.
    Parameters:
    - x (np.ndarray): The input tensor to be padded.
    - l (int): The target length for the first dimension (axis 0).
    - pad_value (int, optional): The value to pad with. Default is 0.
    Returns:
    - np.ndarray: The padded tensor.
    """
    padlen = _calculate_padding_length(x, l)
    pad = _create_padding_array(x, padlen)
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate_flair(sample_dict, pad_value=0):
    """
    Collate function for batching and padding. 
    Pads only the relevant SENTINEL fields and keeps other fields unchanged.
    Args:
    - sample_dict (list): List of dictionaries where each dictionary represents a sample.
    - pad_value (int): The value used for padding tensors.
    Returns:
    - dict: A dictionary with padded tensors and original string keys.
    """

    TO_PAD_KEYS = [
        'SENTINEL2_TS', 'SENTINEL2_DATES',
        'SENTINEL1-ASC_TS', 'SENTINEL1-ASC_DATES',
        'SENTINEL1-DESC_TS', 'SENTINEL1-DESC_DATES'
    ]
    
    batch = {}
    
    for key in sample_dict[0].keys():
        if key in TO_PAD_KEYS:
            data = [i[key] for i in sample_dict]

            if all(len(e) == 0 for e in data):
                batch[key] = torch.empty((len(data), 0))  
                continue

            sizes = [e.shape[0] for e in data if len(e) > 0]
            max_size = max(sizes) if sizes else 0

            padded_data = [
                pad_tensor(d, max_size, pad_value=pad_value) if len(d) > 0 else torch.zeros((max_size,), dtype=d.dtype) 
                for d in data
            ]
            batch[key] = torch.stack(padded_data, dim=0)

        elif isinstance(sample_dict[0][key], torch.Tensor):
            batch[key] = torch.stack([i[key] for i in sample_dict], dim=0)
        else:
            batch[key] = [i[key] for i in sample_dict]

    return batch