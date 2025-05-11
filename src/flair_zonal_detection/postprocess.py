import os
import numpy as np
from shapely.geometry import box, mapping
from scipy.special import softmax
from rasterio.shutil import copy as rio_copy


def convert(img: np.ndarray, img_type: str) -> np.ndarray:
    """
    Convert model output logits to either softmax probabilities (uint8)
    or argmax class predictions.

    Args:
        img: Array of logits with shape (C, H, W)
        img_type: 'class_prob' or 'argmax'

    Returns:
        np.ndarray: Converted output as uint8 or single-channel index map
    """
    if img_type == "class_prob":
        if img.ndim != 3:
            raise ValueError("Expected logits with shape (C, H, W)")
        img = softmax(img, axis=0)  # Apply softmax across classes
        return np.round(img * 255).astype(np.uint8)

    elif img_type == "argmax":
        prediction = np.argmax(img, axis=0)
        return np.expand_dims(prediction.astype(np.uint8), axis=0)

    else:
        raise ValueError(f"Unknown output type: {img_type}")


def convert_to_cog(input_path: str, output_path: str) -> None:
    """
    Convert a GeoTIFF file to Cloud Optimized GeoTIFF (COG) format.

    Args:
        input_path: Path to the input GeoTIFF
        output_path: Path to save the COG
    """
    cog_profile = {
        'driver': 'COG',
        'compress': 'LZW',
        'blocksize': 512,
        'overview_resampling': 'nearest',
        'tiled': True
    }

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rio_copy(input_path, output_path, **cog_profile)
    os.remove(input_path)


def create_polygon_from_bounds(x_min: float, x_max: float, y_min: float, y_max: float) -> dict:
    """
    Create a GeoJSON-style polygon from bounding box coordinates.

    Args:
        x_min: Minimum X (left)
        x_max: Maximum X (right)
        y_min: Minimum Y (bottom)
        y_max: Maximum Y (top)

    Returns:
        GeoJSON-like polygon dictionary
    """
    return mapping(box(x_min, y_max, x_max, y_min))
