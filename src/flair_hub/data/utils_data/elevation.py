import numpy as np

def calc_elevation(arr: np.ndarray) -> np.ndarray:
    """
    Calculates the elevation difference between two input channels.
    Args:
        arr (np.ndarray): Input array where the first channel is elevation and the second is baseline.
    Returns:
        np.ndarray: Array containing the elevation difference with shape (1, height, width).
    """
    elev = arr[0] - arr[1]
    return elev[np.newaxis, :, :]
