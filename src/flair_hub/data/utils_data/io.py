import rasterio
import numpy as np

def read_patch(raster_file: str, channels: list = None) -> np.ndarray:
    """
    Reads patch data from a raster file.

    Args:
        raster_file (str): Path to the raster file.
        channels (list, optional): List of channel indices to read. If None, reads all channels.

    Returns:
        np.ndarray: The extracted patch data.
    """
    with rasterio.open(raster_file) as src_img:
        array = src_img.read(channels) if channels else src_img.read()
    return array
