import numpy as np

def reshape_label_ohe(arr: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Converts a label array into one-hot-encoded format.

    Args:
        arr (np.ndarray): Input label array.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot-encoded label array with shape (num_classes, ...).
    """
    if arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return np.stack([arr == i for i in range(num_classes)], axis=0)
