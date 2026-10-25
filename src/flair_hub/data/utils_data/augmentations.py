import hashlib

import numpy as np
import torch
import cv2
from torch.nn import functional as F


def _resize_spatial_numpy(
    x: np.ndarray,
    new_h: int,
    new_w: int,
    is_label: bool = False,
) -> np.ndarray:
    """
    Resize spatial dimensions of an array.

    Supported shapes:
        [C, H, W]
        [T, C, H, W]

    Images are resized with bilinear interpolation.
    Labels are resized with nearest-neighbor interpolation.
    """
    original_dtype = x.dtype

    if x.ndim == 3:
        xt = torch.from_numpy(x).unsqueeze(0).float()

    elif x.ndim == 4:
        xt = torch.from_numpy(x).float()

    else:
        return x

    if is_label:
        yt = F.interpolate(xt,size=(new_h, new_w),mode="nearest",
                           )
    else:
        yt = F.interpolate(xt,size=(new_h, new_w),mode="bilinear",align_corners=False,
                           )

    if x.ndim == 3:
        y = yt.squeeze(0).numpy()
    else:
        y = yt.numpy()

    return y.astype(original_dtype, copy=False)


def _pad_to_size_numpy(
    x: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float = 0,
) -> np.ndarray:
    """
    Pad spatial dimensions of an array.

    Supported shapes:
        [C, H, W]
        [T, C, H, W]
    """
    if x.ndim == 3:
        _, h, w = x.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        return np.pad(
            x,
            pad_width=((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )

    if x.ndim == 4:
        _, _, h, w = x.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        return np.pad(
            x,
            pad_width=((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )

    return x


def _crop_spatial_numpy(
    x: np.ndarray,
    y0: int,
    x0: int,
    crop_h: int,
    crop_w: int,
) -> np.ndarray:
    """
    Crop spatial dimensions of an array.

    Supported shapes:
        [C, H, W]
        [T, C, H, W]
    """
    if x.ndim == 3:
        return x[:, y0 : y0 + crop_h, x0 : x0 + crop_w]

    if x.ndim == 4:
        return x[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w]

    return x


def _hflip_numpy(x: np.ndarray) -> np.ndarray:
    """
    Horizontally flip spatial dimensions.
    """
    if x.ndim == 3:
        return x[:, :, ::-1].copy()

    if x.ndim == 4:
        return x[:, :, :, ::-1].copy()

    return x


def _vflip_numpy(x: np.ndarray) -> np.ndarray:
    """
    Vertically flip spatial dimensions.
    """
    if x.ndim == 3:
        return x[:, ::-1, :].copy()

    if x.ndim == 4:
        return x[:, :, ::-1, :].copy()

    return x


def _rot90_numpy(x: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate spatial dimensions by k * 90 degrees.

    Supported shapes:
        [C, H, W]
        [T, C, H, W]
    """
    if k == 0:
        return x

    if x.ndim == 3:
        return np.rot90(x, k=k, axes=(1, 2)).copy()

    if x.ndim == 4:
        return np.rot90(x, k=k, axes=(2, 3)).copy()

    return x


def _normalized_radiometric_jitter_numpy(
    x: np.ndarray,
    rng: np.random.Generator,
    gain_range: tuple[float, float] = (0.85, 1.15),
    gain_prob: float = 0.7,
    offset_range: tuple[float, float] = (-0.10, 0.10),
    offset_prob: float = 0.5,
    noise_std_range: tuple[float, float] = (0.0, 0.04),
    noise_prob: float = 0.3,
) -> np.ndarray:
    """
    Apply mild radiometric jitter to already-normalized imagery.

    Supported shapes:
        [C, H, W]
        [T, C, H, W]

    Gain and offset are sampled per channel and shared spatially. Noise is
    sampled per pixel. The operation is intentionally not clipped because
    custom-normalized tensors can legitimately exceed a fixed value range.
    """
    if x.ndim not in (3, 4):
        return x

    original_dtype = x.dtype
    y = x.astype(np.float32, copy=True)

    if x.ndim == 3:
        channel_shape = (x.shape[0], 1, 1)
    else:
        channel_shape = (1, x.shape[1], 1, 1)

    if rng.random() < gain_prob:
        gain = rng.uniform(
            gain_range[0],
            gain_range[1],
            size=channel_shape,
        ).astype(np.float32)
        y *= gain

    if rng.random() < offset_prob:
        offset = rng.uniform(
            offset_range[0],
            offset_range[1],
            size=channel_shape,
        ).astype(np.float32)
        y += offset

    if rng.random() < noise_prob:
        noise_std = rng.uniform(noise_std_range[0], noise_std_range[1])
        noise = rng.normal(0.0, noise_std, size=y.shape).astype(np.float32)
        y += noise

    return y.astype(original_dtype, copy=False)


def apply_normalized_radiometric_jitter_numpy(
    batch: dict,
    input_keys: list[str],
    radiometric_keys: tuple[str, ...] = (
        "AERIAL_RGBI",
        "AERIAL-RLT_PAN",
        "SPOT_RGBI",
    ),
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Apply normalized-domain radiometric jitter to optical inputs only.
    """
    if rng is None:
        rng = np.random.default_rng()

    enabled_radiometric_keys = set(input_keys).intersection(radiometric_keys)

    for key in enabled_radiometric_keys:
        if (
            key in batch
            and isinstance(batch[key], np.ndarray)
            and batch[key].ndim in (3, 4)
        ):
            batch[key] = _normalized_radiometric_jitter_numpy(batch[key], rng)

    return batch


def apply_scale_crop_flip_rotate_numpy(
    batch: dict,
    input_keys: list[str],
    label_keys: list[str],
    scale_range: tuple[float, float] = (0.8, 1.5),
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rot90_prob: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Apply geometric augmentations while preserving each array's original spatial size.

    Supported input shapes:
        [C, H, W]
        [T, C, H, W]

    Shared random parameters are used across all spatial inputs and labels:
        - same scale factor
        - same relative crop position
        - same horizontal flip decision
        - same vertical flip decision
        - same 90-degree rotation decision

    This keeps modalities spatially aligned while allowing each modality to keep
    its own native patch size, e.g. 512x512 for aerial imagery and 10x10 for
    Sentinel data.
    """
    if rng is None:
        rng = np.random.default_rng()

    label_key_set = set(label_keys)

    spatial_keys = [
        key
        for key in input_keys + label_keys
        if (
            key in batch
            and isinstance(batch[key], np.ndarray)
            and batch[key].ndim in (3, 4)
        )
    ]

    if not spatial_keys:
        return batch

    original_sizes = {
        key: batch[key].shape[-2:]
        for key in spatial_keys
    }

    scale = rng.uniform(scale_range[0], scale_range[1])

    crop_y_rel = rng.random()
    crop_x_rel = rng.random()

    do_hflip = rng.random() < hflip_prob
    do_vflip = rng.random() < vflip_prob
    do_rot90 = rng.random() < rot90_prob
    k_rot = int(rng.integers(1, 4)) if do_rot90 else 0

    for key in spatial_keys:

        orig_h, orig_w = original_sizes[key]

        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))

        batch[key] = _resize_spatial_numpy(
            batch[key],
            new_h=new_h,
            new_w=new_w,
            is_label=key in label_key_set,
        )

        batch[key] = _pad_to_size_numpy(
            batch[key],
            target_h=orig_h,
            target_w=orig_w,
            pad_value=-100 if key in label_key_set else 0,
        )

        current_h, current_w = batch[key].shape[-2:]
        max_y0 = current_h - orig_h
        max_x0 = current_w - orig_w

        y0 = int(round(crop_y_rel * max_y0)) if max_y0 > 0 else 0
        x0 = int(round(crop_x_rel * max_x0)) if max_x0 > 0 else 0

        batch[key] = _crop_spatial_numpy(
            batch[key],
            y0=y0,
            x0=x0,
            crop_h=orig_h,
            crop_w=orig_w,
        )

    if do_hflip:
        for key in spatial_keys:
            batch[key] = _hflip_numpy(batch[key])

    if do_vflip:
        for key in spatial_keys:
            batch[key] = _vflip_numpy(batch[key])

    if do_rot90:
        for key in spatial_keys:
            batch[key] = _rot90_numpy(batch[key], k=k_rot)

    return batch


def apply_numpy_augmentations(
    batch: dict,
    input_keys: list[str],
    label_keys: list[str],
) -> dict:
    """
    Apply the default NumPy/PyTorch augmentation pipeline.
    """
    rng = np.random.default_rng()

    batch = apply_scale_crop_flip_rotate_numpy(
        batch,
        input_keys=input_keys,
        label_keys=label_keys,
        scale_range=(1.0, 1.75),
        hflip_prob=0.5,
        vflip_prob=0.5,
        rot90_prob=0.5,
        rng=rng,
    )

    return apply_normalized_radiometric_jitter_numpy(
        batch,
        input_keys=input_keys,
        rng=rng,
    )


def _production_crop_bbox(
    image: np.ndarray,
    crop_size: tuple[int, int],
    rng,
) -> tuple[int, int, int, int]:
    """Sample a top-left position for random spatial cropping."""
    crop_h, crop_w = crop_size
    height, width = image.shape[-2:]
    margin_h = max(height - crop_h, 0)
    margin_w = max(width - crop_w, 0)
    y0 = int(rng.randint(0, margin_h + 1))
    x0 = int(rng.randint(0, margin_w + 1))
    return y0, x0, min(crop_h, height), min(crop_w, width)


def _production_random_crop(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: tuple[int, int],
    cat_max_ratio: float,
    ignore_index: int,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop with a ten-attempt dominant-class constraint."""
    crop_bbox = _production_random_crop_bbox(
        image,
        label,
        crop_size,
        cat_max_ratio,
        ignore_index,
        rng,
    )

    y0, x0, crop_h, crop_w = crop_bbox
    return (
        image[..., y0:y0 + crop_h, x0:x0 + crop_w],
        label[..., y0:y0 + crop_h, x0:x0 + crop_w],
    )


def _production_random_crop_bbox(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: tuple[int, int],
    cat_max_ratio: float,
    ignore_index: int,
    rng,
) -> tuple[int, int, int, int]:
    """Sample a bounding box subject to a dominant-class constraint.

    The sampler draws an initial crop and, after every
    rejected attempt, samples the candidate for the next iteration. If all ten
    checked candidates are rejected, the final newly sampled candidate is used
    without another class-ratio check. The final candidate is returned if no candidate satisfies the constraint.
    """
    crop_bbox = _production_crop_bbox(image, crop_size, rng)
    for _ in range(10):
        y0, x0, crop_h, crop_w = crop_bbox
        label_crop = label[..., y0:y0 + crop_h, x0:x0 + crop_w]
        labels, counts = np.unique(label_crop, return_counts=True)
        counts = counts[labels != ignore_index]
        if (
            len(counts) > 1
            and counts.max() / counts.sum() < cat_max_ratio
        ):
            break
        crop_bbox = _production_crop_bbox(image, crop_size, rng)

    return crop_bbox


def _production_convert(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Match the uint8 clipping used by the bundled PhotoMetricDistortion."""
    converted = image.astype(np.float32) * alpha + beta
    return np.clip(converted, 0, 255).astype(np.uint8)


def _production_photometric_distortion(
    image: np.ndarray,
    rng,
    *,
    brightness_delta: float = 32,
    contrast_range: tuple[float, float] = (0.5, 1.5),
    saturation_range: tuple[float, float] = (0.5, 1.5),
    hue_delta: int = 18,
) -> np.ndarray:
    """Apply brightness/contrast/saturation/hue to a declared pseudo-RGB input."""
    image_hwc = np.moveaxis(image, 0, -1)

    if int(rng.randint(2)):
        image_hwc = _production_convert(
            image_hwc,
            beta=float(rng.uniform(-brightness_delta, brightness_delta)),
        )

    contrast_first = int(rng.randint(2)) == 1
    if contrast_first and int(rng.randint(2)):
        image_hwc = _production_convert(
            image_hwc,
            alpha=float(rng.uniform(*contrast_range)),
        )

    if int(rng.randint(2)):
        hsv = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = _production_convert(
            hsv[..., 1],
            alpha=float(rng.uniform(*saturation_range)),
        )
        image_hwc = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if int(rng.randint(2)):
        hsv = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2HSV)
        sampled_hue_delta = int(rng.randint(-hue_delta, hue_delta))
        hsv[..., 0] = (
            hsv[..., 0].astype(np.int16) + sampled_hue_delta
        ) % 180
        image_hwc = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if not contrast_first and int(rng.randint(2)):
        image_hwc = _production_convert(
            image_hwc,
            alpha=float(rng.uniform(*contrast_range)),
        )

    return np.moveaxis(image_hwc, -1, 0)


def _production_resize(
    array: np.ndarray,
    height: int,
    width: int,
    *,
    is_label: bool,
) -> np.ndarray:
    """Resize CHW data with the OpenCV interpolation used by MMCV."""
    interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    leading_shape = array.shape[:-2]
    planes = array.reshape((-1, *array.shape[-2:]))
    channels = [
        cv2.resize(
            channel,
            (width, height),
            interpolation=interpolation,
        )
        for channel in planes
    ]
    return np.stack(channels, axis=0).reshape(
        (*leading_shape, height, width)
    ).astype(array.dtype, copy=False)


def _production_rotate(
    array: np.ndarray,
    angle: float,
    *,
    is_label: bool,
) -> np.ndarray:
    """Rotate without expanding bounds, matching MMCV ``auto_bound=False``."""
    height, width = array.shape[-2:]
    center = ((width - 1) * 0.5, (height - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    border_value = -100 if is_label else 0

    leading_shape = array.shape[:-2]
    planes = array.reshape((-1, height, width))
    channels = []
    for channel in planes:
        channels.append(
            cv2.warpAffine(
                channel,
                matrix,
                (width, height),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
        )
    return np.stack(channels, axis=0).reshape(
        (*leading_shape, height, width)
    ).astype(array.dtype, copy=False)


def _apply_production_radiometry(
    value: np.ndarray,
    modality: str,
    radiometry: dict,
    rng,
) -> np.ndarray:
    """Apply radiometry with channel positions interpreted as pseudo-RGB."""
    if radiometry.get("mode", "hsv") == "gain_offset":
        channels = value.shape[0] if value.ndim == 3 else value.shape[1]
        shape = (channels, 1, 1) if value.ndim == 3 else (1, channels, 1, 1)
        gain = rng.uniform(*radiometry.get("gain_range", [0.9, 1.1]), size=shape)
        offset = rng.uniform(*radiometry.get("offset_range", [0.0, 0.0]), size=shape)
        return (value.astype(np.float32) * gain + offset).astype(np.float32)
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 255:
        raise ValueError(
            f"{modality}: HSV requires values in [0,255]; use gain_offset for native radiometry"
        )
    parameters = {
        "brightness_delta": float(radiometry.get("brightness_delta", 32)),
        "contrast_range": tuple(radiometry.get("contrast_range", [0.5, 1.5])),
        "saturation_range": tuple(radiometry.get("saturation_range", [0.5, 1.5])),
        "hue_delta": int(radiometry.get("hue_delta", 18)),
    }
    if value.ndim == 3 and value.shape[0] == 3:
        return _production_photometric_distortion(value, rng, **parameters)
    if value.ndim == 4 and value.shape[1] == 3:
        return np.stack([
            _production_photometric_distortion(frame, rng, **parameters)
            for frame in value
        ])
    raise ValueError(
        f"Configured HSV radiometry for {modality} requires exactly three channels"
    )


def _fork_rng_without_advancing(rng, namespace: str) -> np.random.RandomState:
    """Derive a stable modality RNG without perturbing the shared geometry RNG.

    Extra modalities did not exist in the historical aerial recipe. Giving
    each one a deterministic stream derived from the current worker RNG state
    prevents enabling an extra radiometric transform from changing the aerial
    transform, the shared rotation, or every subsequent training sample.
    """
    digest = hashlib.sha256(namespace.encode("utf-8"))
    if hasattr(rng, "get_state"):
        state = rng.get_state()
        digest.update(str(state[0]).encode("ascii"))
        digest.update(np.asarray(state[1], dtype=np.uint32).tobytes())
        digest.update(repr(state[2:]).encode("ascii"))
    elif hasattr(rng, "bit_generator"):
        digest.update(repr(rng.bit_generator.state).encode("utf-8"))
    else:
        raise TypeError(
            "Configured augmentations require a NumPy RandomState, Generator, "
            "or numpy.random module"
        )
    seed = int.from_bytes(digest.digest()[:4], "little")
    return np.random.RandomState(seed)


def apply_configured_augmentations(
    batch: dict,
    input_keys: list[str],
    label_keys: list[str],
    config: dict,
    *,
    rng=None,
) -> dict:
    """Apply an explicit, synchronized augmentation configuration.

    Geometry is sampled once and transferred through relative crop positions,
    so modalities can retain different native resolutions. The same transform
    is applied to every temporal frame and every segmentation task. The default
    aerial recipe uses a synchronized random sequence;
    extra modalities extend that transform without perturbing it. Radiometry is
    opt-in per modality and never touches labels.
    """
    if rng is None:
        rng = np.random

    geometry = config.get("geometry", {})
    label_set = set(label_keys)
    spatial_keys = [
        key for key in input_keys + label_keys
        if key in batch
        and isinstance(batch[key], np.ndarray)
        and batch[key].ndim in (3, 4)
    ]
    if not spatial_keys:
        return batch

    original_sizes = {key: batch[key].shape[-2:] for key in spatial_keys}
    scale_low, scale_high = geometry.get("scale_range", [1.0, 1.0])
    scale = float(rng.uniform(scale_low, scale_high))
    for key in spatial_keys:
        height, width = original_sizes[key]
        batch[key] = _production_resize(
            batch[key],
            max(1, int(height * scale)),
            max(1, int(width * scale)),
            is_label=key in label_set,
        )

    reference_key = geometry.get("cropping_reference_task")
    if reference_key not in label_set or reference_key not in batch:
        reference_key = next((key for key in label_keys if key in batch), spatial_keys[0])
    reference = batch[reference_key]
    target_h, target_w = original_sizes[reference_key]

    cat_max_ratio = float(geometry.get("category_max_ratio", 1.0))
    ignore_index = -100
    ref_h, ref_w = reference.shape[-2:]
    if reference_key in label_set:
        y0, x0, _, _ = _production_random_crop_bbox(
            reference,
            reference,
            (target_h, target_w),
            cat_max_ratio,
            ignore_index,
            rng,
        )
    else:
        y0, x0, _, _ = _production_crop_bbox(
            reference, (target_h, target_w), rng
        )
    ref_margin_h = max(ref_h - target_h, 0)
    ref_margin_w = max(ref_w - target_w, 0)
    y_rel = y0 / ref_margin_h if ref_margin_h else 0.0
    x_rel = x0 / ref_margin_w if ref_margin_w else 0.0

    for key in spatial_keys:
        current_h, current_w = batch[key].shape[-2:]
        crop_h, crop_w = original_sizes[key]
        margin_h = max(current_h - crop_h, 0)
        margin_w = max(current_w - crop_w, 0)
        key_y0 = (
            y0 if margin_h == ref_margin_h
            else int(round(y_rel * margin_h)) if margin_h else 0
        )
        key_x0 = (
            x0 if margin_w == ref_margin_w
            else int(round(x_rel * margin_w)) if margin_w else 0
        )
        batch[key] = _crop_spatial_numpy(
            batch[key],
            key_y0,
            key_x0,
            min(crop_h, current_h),
            min(crop_w, current_w),
        )

    hflip_probability = float(geometry.get("horizontal_flip_probability", 0.0))
    vflip_probability = float(geometry.get("vertical_flip_probability", 0.0))
    if hflip_probability == 0.0 and vflip_probability == 0.0:
        if float(rng.random()) < 0.5:
            for key in spatial_keys:
                batch[key] = _hflip_numpy(_hflip_numpy(batch[key]))
    else:
        if float(rng.random()) < hflip_probability:
            for key in spatial_keys:
                batch[key] = _hflip_numpy(batch[key])
        if float(rng.random()) < vflip_probability:
            for key in spatial_keys:
                batch[key] = _vflip_numpy(batch[key])

    radiometry_items = [
        (key, radiometry)
        for key, radiometry in config.get("radiometry", {}).items()
        if radiometry.get("enabled", False) and key in batch
    ]
    aerial_radiometry = next(
        (item for item in radiometry_items if item[0] == "AERIAL_RGBI"),
        None,
    )
    if aerial_radiometry is not None:
        key, radiometry = aerial_radiometry
        batch[key] = _apply_production_radiometry(
            batch[key], key, radiometry, rng
        )

    rotate = float(rng.random()) < float(geometry.get("rotation_probability", 0.0))
    angle_low, angle_high = geometry.get("rotation_degrees", [0.0, 0.0])
    angle = float(rng.uniform(angle_low, angle_high))

    for key, radiometry in radiometry_items:
        if key == "AERIAL_RGBI":
            continue
        modality_rng = _fork_rng_without_advancing(rng, f"radiometry:{key}")
        batch[key] = _apply_production_radiometry(
            batch[key], key, radiometry, modality_rng
        )

    if rotate:
        for key in spatial_keys:
            batch[key] = _production_rotate(
                batch[key], angle, is_label=key in label_set
            )
    return batch


def apply_aerial_production_augmentations(
    batch: dict,
    input_keys: list[str],
    label_keys: list[str],
    *,
    rng=None,
) -> dict:
    """Apply the geometric and photometric production training pipeline.

    The final Normalize and Pad operations intentionally remain in the dataset:
    the source pipeline normalizes first and pads normalized images with zero.
    """
    if rng is None:
        rng = np.random

    if len(input_keys) != 1 or len(label_keys) != 1:
        raise ValueError(
            "The aerial production augmentation profile supports the single-aerial, "
            "single-COSIA-task workflow only"
        )

    image_key = input_keys[0]
    label_key = label_keys[0]
    image = batch[image_key]
    label = batch[label_key]

    scale = float(rng.uniform(0.5, 2.0))
    height, width = image.shape[-2:]
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))
    image = _production_resize(
        image, new_h, new_w, is_label=False
    )
    label = _production_resize(
        label, new_h, new_w, is_label=True
    )

    image, label = _production_random_crop(
        image,
        label,
        crop_size=(512, 512),
        cat_max_ratio=0.75,
        ignore_index=-100,
        rng=rng,
    )


    if float(rng.random()) < 0.5:
        image, label = _hflip_numpy(image), _hflip_numpy(label)
        image, label = _hflip_numpy(image), _hflip_numpy(label)

    image = _production_photometric_distortion(image, rng)

    rotate = float(rng.random()) < 0.5
    angle = float(rng.uniform(-90.0, 90.0))
    if rotate:
        image = _production_rotate(image, angle, is_label=False)
        label = _production_rotate(label, angle, is_label=True)

    batch[image_key] = image
    batch[label_key] = label
    return batch


def pad_aerial_production_sample(
    batch: dict,
    input_keys: list[str],
    label_keys: list[str],
    size: tuple[int, int] = (512, 512),
) -> dict:
    """Pad after normalization with image value 0 and label value -100."""
    for key in input_keys:
        if key in batch:
            batch[key] = _pad_to_size_numpy(
                batch[key], size[0], size[1], pad_value=0
            )
    for key in label_keys:
        if key in batch:
            batch[key] = _pad_to_size_numpy(
                batch[key], size[0], size[1], pad_value=-100
            )
    return batch


def pad_sample_to_original_sizes(
    batch: dict,
    target_sizes: dict[str, tuple[int, int]],
    label_keys: list[str],
) -> dict:
    """Restore native spatial sizes after scale/crop augmentation."""
    label_set = set(label_keys)
    for key, (height, width) in target_sizes.items():
        if key in batch:
            batch[key] = _pad_to_size_numpy(
                batch[key],
                height,
                width,
                pad_value=-100 if key in label_set else 0,
            )
    return batch
