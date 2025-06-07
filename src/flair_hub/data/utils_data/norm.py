import sys
import numpy as np

from skimage import img_as_float


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
