import os
import sys
import torch
import rasterio
import traceback
import time
import datetime
import numpy as np

from typing import Dict, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio.transform import from_origin
from scipy.ndimage import zoom

from src.flair_zonal_detection.config import (
    load_config,
    validate_config,
    config_recap_1, config_recap_2
)
from src.flair_hub.utils.messaging import Logger
from src.flair_zonal_detection.dataset import MultiModalSlicedDataset
from src.flair_zonal_detection.postprocess import (
    convert,
    convert_to_cog
)
from src.flair_zonal_detection.model_utils import (
    build_inference_model,
    compute_patch_sizes
)
from src.flair_zonal_detection.slicing import generate_patches_from_reference

from scipy.ndimage import zoom

def prep_config(config_path: str) -> Dict:
    """
    Load and validate configuration, initialize logging and device.
    """
    config = load_config(config_path)
    log_filename = os.path.join(
        config['output_path'],
        f"{config['output_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    sys.stdout = Logger(filename=log_filename)
    print(f"\n[LOGGER] Writing logs to: {log_filename}")

    validate_config(config)
    config_recap_1(config)
    config = initialize_geometry_and_resolutions(config)
    config_recap_2(config)

    config['device'] = torch.device("cuda" if config.get("use_gpu", torch.cuda.is_available()) else "cpu")
    config['output_type'] = config.get("output_type", "argmax")
    return config


def initialize_geometry_and_resolutions(config: Dict) -> Dict:
    """
    Determine bounding box and resolution consistency across all active modalities.
    Sets:
        - config['reference_resolution']
        - config['modality_resolutions']
        - config['image_bounds']
    Validates:
        - All bounds match (warn or fail if not)
    """
    modalities = config['modalities']
    active_mods = [mod for mod, is_active in modalities['inputs'].items() if is_active]

    resolutions = {}
    bounds = []

    for mod in active_mods:
        path = modalities[mod]['input_img_path']
        with rasterio.open(path) as src:
            resolutions[mod] = round(src.res[0], 5)
            bounds.append((mod, src.bounds))
            
            if 'image_shape_px' not in config:
                config['image_shape_px'] = {
                    'height': src.height,
                    'width': src.width
                }

    # Check bounds match
    ref_mod, ref_bounds = bounds[0]
    for mod, b in bounds[1:]:
        if not np.allclose(b, ref_bounds, atol=1e-2):
            raise ValueError(
                f"[✗] Bounds mismatch between '{ref_mod}' and '{mod}':\n"
                f"  {ref_mod}: {ref_bounds}\n"
                f"  {mod}: {b}"
            )

    # Choose reference modality based on coarsest resolution (largest m/px)
    ref_mod, reference_resolution = min(resolutions.items(), key=lambda x: x[1])
    config['reference_modality'] = ref_mod
    config['reference_resolution'] = reference_resolution

    config['modality_resolutions'] = resolutions
    config['image_bounds'] = {
        'left': ref_bounds.left,
        'bottom': ref_bounds.bottom,
        'right': ref_bounds.right,
        'top': ref_bounds.top
    }

    tile_size_m = config['img_pixels_detection'] * reference_resolution
    margin_size_m = config['margin'] * reference_resolution
    config['tile_size_m'] = round(tile_size_m, 2)
    config['margin_size_m'] = round(margin_size_m, 2)

    return config



def prep_dataset(config: Dict, tiles_gdf, patch_sizes: Dict[str, int]) -> MultiModalSlicedDataset:
    """
    Prepare the dataset object from config and sliced patches.
    """
    active_mods = [m for m, active in config['modalities']['inputs'].items() if active]
    modality_cfgs = {m: config['modalities'][m] for m in active_mods}

    config['labels'] = [t['name'] for t in config['tasks'] if t['active']]
    config['labels_configs'] = {
        t['name']: {'value_name': t['class_names']} for t in config['tasks'] if t['active']
    }

    return MultiModalSlicedDataset(
        dataframe=tiles_gdf,
        modality_cfgs=modality_cfgs,
        patch_size_dict=patch_sizes,
        ref_date_str=config['multitemp_model_ref_date'],
        modalities_config=config
    )


def init_outputs(config: Dict, ref_img: DatasetReader) -> Tuple[Dict[str, DatasetReader], Dict[str, str]]:
    """
    Initialize output raster files per task. Adjusts dimensions and transform if resolution differs.
    """
    output_files = {}
    temp_paths = {}
    output_type = config['output_type']
    ref_res = config['reference_resolution']
    out_res = config.get("output_px_meters", ref_res)
    image_bounds = config['image_bounds']
    needs_rescale = abs(ref_res - out_res) > 1e-6

    for task in config['tasks']:
        if not task['active']:
            continue

        num_classes = len(task['class_names'])
        suffix = 'argmax' if output_type == 'argmax' else 'class-prob'
        out_path = os.path.join(
            config['output_path'],
            f"{config['output_name']}_{task['name']}_{suffix}.tif"
        )

        if not needs_rescale:
            # Use reference image profile directly
            profile = ref_img.profile.copy()
            profile.update({
                "count": num_classes if output_type == "class_prob" else 1,
                "dtype": "uint8",
                "compress": "lzw"
            })
        else:
            # Adjust height, width and transform based on new resolution
            out_height = int(round((image_bounds['top'] - image_bounds['bottom']) / out_res))
            out_width = int(round((image_bounds['right'] - image_bounds['left']) / out_res))
            transform = from_origin(image_bounds['left'], image_bounds['top'], out_res, out_res)

            profile = {
                "driver": "GTiff",
                "height": out_height,
                "width": out_width,
                "count": num_classes if output_type == "class_prob" else 1,
                "dtype": "uint8",
                "crs": ref_img.crs,
                "transform": transform,
                "compress": "lzw"
            }

        output_files[task['name']] = rasterio.open(out_path, 'w', **profile)
        temp_paths[task['name']] = out_path

    return output_files, temp_paths



def resample_prediction(prediction: np.ndarray, scale: float) -> np.ndarray:
    """
    Resample prediction using nearest-neighbor zoom.
    
    Handles both:
    - (H, W) for argmax
    - (C, H, W) for logits/class-probabilities
    """
    if prediction.ndim == 2: 
        return zoom(prediction, zoom=scale, order=0)
    elif prediction.ndim == 3:
        c, h, w = prediction.shape
        return zoom(prediction, zoom=(1, scale, scale), order=0)
    else:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")


def inference_and_write(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tiles_gdf,
    config: Dict,
    output_files: Dict[str, DatasetReader],
    ref_img: DatasetReader
) -> None:
    """
    Run model inference and write predictions to raster files.
    Supports resampling logits to output_px_meters if different from reference_resolution.
    """
    device = config['device']
    margin_px = config['margin']
    tile_size = config['img_pixels_detection']
    output_type = config['output_type']
    ref_res = config['reference_resolution']
    out_res = config.get('output_px_meters', ref_res)  # fallback to ref_res if not set
    needs_rescale = abs(ref_res - out_res) > 1e-6
    image_bounds = config['image_bounds']

    print("\n[ ] Starting inference and writing raster tiles...\n")

    for batch in tqdm(dataloader, file=sys.stdout):
        inputs = {
            mod: batch[mod].to(device)
            for mod in batch if mod not in ['index'] and not mod.endswith('_DATES')
        }
        for mod in batch:
            if mod.endswith('_DATES'):
                inputs[mod] = batch[mod].to(device)

        indices = batch['index'].cpu().numpy().flatten()
        rows = tiles_gdf.iloc[indices]

        with torch.no_grad():
            logits_tasks, _ = model(inputs)

        for task_name, logits in logits_tasks.items():
            logits = logits.cpu().numpy()

            for i, idx in enumerate(indices):
                row = rows.iloc[i]

                logit_patch = logits[i, :, margin_px:tile_size - margin_px, margin_px:tile_size - margin_px]
                res_ref = config["reference_resolution"]
                res_out = config.get("output_px_meters", res_ref)
                needs_rescale = abs(res_out - res_ref) > 1e-6
                scale = res_ref / res_out if needs_rescale else 1.0

                if output_type == "argmax":
                    prediction = convert(logit_patch, "argmax")  # shape: (H, W)
                    if needs_rescale:
                        prediction = resample_prediction(prediction, scale)
                else:
                    if needs_rescale:
                        logit_patch = resample_prediction(logit_patch, scale)
                    prediction = convert(logit_patch, output_type)  # (C, H, W)

                # Get top-left corner in output raster
                left = row['left']
                top = row['top']
                left_px = int(round((left - image_bounds['left']) / out_res))
                top_px  = int(round((image_bounds['top'] - top) / out_res))

                # Get prediction size
                height_px = prediction.shape[-2]
                width_px  = prediction.shape[-1]

                # Output raster dimensions
                img_height = int(round((image_bounds['top'] - image_bounds['bottom']) / out_res))
                img_width  = int(round((image_bounds['right'] - image_bounds['left']) / out_res))

                # Clip
                if top_px + height_px > img_height:
                    height_px = img_height - top_px
                if left_px + width_px > img_width:
                    width_px = img_width - left_px

                if height_px <= 0 or width_px <= 0:
                    print(f"[!] Skipping tile {row['id']} — window out of bounds.")
                    continue

                # Crop prediction if needed
                prediction = prediction[..., :height_px, :width_px]
                window = Window(col_off=left_px, row_off=top_px, width=width_px, height=height_px)

                # Write
                if output_type == "argmax":
                    output_files[task_name].write(prediction[0], 1, window=window)
                else:
                    for c in range(prediction.shape[0]):
                        output_files[task_name].write(prediction[c], c + 1, window=window)

    for dst in output_files.values():
        dst.close()





def postpro_outputs(temp_paths: Dict[str, str], config: Dict) -> None:
    """
    Convert output rasters to Cloud Optimized GeoTIFFs (COG) if requested.
    """
    if config.get("cog_conversion", False):
        for task_name, temp_path in temp_paths.items():
            cog_path = temp_path.replace(".tif", "_COG.tif")
            convert_to_cog(temp_path, cog_path)
            print(f"\n[✓] Converted to COG: {cog_path}")


def run_inference(config_path: str) -> None:
    """
    Main entry point to run inference from a config file.
    """
    try:
        start_total = time.time()
        config = prep_config(config_path)
        start_slice = time.time()
        tiles_gdf = generate_patches_from_reference(config)
        print(f"[✓] Sliced into {len(tiles_gdf)} tiles in {time.time() - start_slice:.2f}s")

        start_model = time.time()
        patch_sizes = compute_patch_sizes(config)

        model = build_inference_model(config, patch_sizes).to(config['device'])
        print(f"[✓] Loaded model and checkpoint in {time.time() - start_model:.2f}s")

        dataset = prep_dataset(config, tiles_gdf, patch_sizes)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_worker'])

        ref_img = rasterio.open(config['modalities'][config['reference_modality']]['input_img_path'])
        output_files, temp_paths = init_outputs(config, ref_img)

        start_infer = time.time()
        inference_and_write(model, dataloader, tiles_gdf, config, output_files, ref_img)
        print(f"[✓] Inference completed in {time.time() - start_infer:.2f}s")

        postpro_outputs(temp_paths, config)
        
        print(f"\n[✓] Total time: {time.time() - start_total:.2f}s")
        print(f"\n[✓] Inference complete. Rasters written to: {list(temp_paths.values())}\n")

    except Exception:
        print("\n[✗] Inference failed with an error:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__