"""Shared training and zonal-inference input preparation."""

from __future__ import annotations

import numpy as np


def reorder_bands(
    array: np.ndarray,
    current_order: list[int],
    input_band_order: list[int],
) -> np.ndarray:
    """Reorder the channel axis to the explicit model input order."""
    if not input_band_order or current_order == input_band_order:
        return np.ascontiguousarray(array)
    try:
        indices = [current_order.index(band) for band in input_band_order]
    except ValueError as exc:
        raise ValueError(
            f"Cannot create input_band_order={input_band_order} from loaded bands {current_order}"
        ) from exc
    channel_axis = 0 if array.ndim == 3 else 1
    return np.ascontiguousarray(np.take(array, indices, axis=channel_axis))


def normalize_array(
    array: np.ndarray,
    normalization_type: str,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Normalize CHW or TCHW data without changing its channel order."""
    normalization_type = str(normalization_type).lower()
    if normalization_type == "without":
        return array.astype(np.float32, copy=False)
    if normalization_type == "scaling":
        return array.astype(np.float32, copy=False) * scale + offset
    if normalization_type != "custom":
        raise ValueError(f"Unsupported normalization type: {normalization_type}")

    mean_array = np.asarray(mean or [], dtype=np.float32)
    std_array = np.asarray(std or [], dtype=np.float32)
    channels = array.shape[0] if array.ndim == 3 else array.shape[1]
    if len(mean_array) != channels or len(std_array) != channels:
        raise ValueError(
            f"Normalization expects {channels} mean/std values, got "
            f"{len(mean_array)} and {len(std_array)}"
        )
    shape = (channels, 1, 1) if array.ndim == 3 else (1, channels, 1, 1)
    return (
        (array.astype(np.float32, copy=False) - mean_array.reshape(shape))
        / std_array.reshape(shape)
    )


def prepare_model_input(
    array: np.ndarray,
    *,
    loaded_band_order: list[int],
    input_band_order: list[int],
    normalization_type: str,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Apply the exact band ordering then normalization used by the model."""
    array = reorder_bands(array, loaded_band_order, input_band_order)
    empty_frames = np.all(array == 0, axis=(1, 2, 3)) if array.ndim == 4 else None
    result = normalize_array(array, normalization_type, mean, std, scale, offset)
    if empty_frames is not None and empty_frames.any():
        result = result.copy()
        result[empty_frames] = 0
    return result
