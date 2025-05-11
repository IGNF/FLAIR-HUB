import os
import numpy as np
from shapely.geometry import box, mapping
from scipy.special import softmax
from rasterio.shutil import copy as rio_copy


def convert(img, img_type):
    if img_type == "class_prob":
        if img.ndim == 3:  # (num_classes, H, W)
            # Apply softmax along the class dimension
            img = softmax(img, axis=0)
        else:
            raise ValueError("Expected logits with shape (C, H, W)")

        # Convert to 0–255 and uint8
        img = np.round(img * 255).astype(np.uint8)
        return img

    elif img_type == "argmax":
        img = np.argmax(img, axis=0)
        return np.expand_dims(img.astype(np.uint8), axis=0)

    else:
        raise ValueError("The output type has not been interpreted.")
    
    
def convert_to_cog(input_path, output_path):
    cog_profile = {
        'driver': 'COG',
        'compress': 'LZW',
        'blocksize': 512,
        'overview_resampling': 'nearest',
        'tiled': True
    }
    rio_copy(input_path, output_path, **cog_profile)
    os.remove(input_path)


def create_polygon_from_bounds(x_min, x_max, y_min, y_max):
    return mapping(box(x_min, y_max, x_max, y_min))