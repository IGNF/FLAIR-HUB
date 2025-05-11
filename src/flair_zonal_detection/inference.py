import os
import sys
import torch
import rasterio
import traceback
import time
import datetime
from typing import Dict, Tuple

from tqdm import tqdm
from torch.utils.data import DataLoader
from rasterio.io import DatasetReader
from rasterio.features import geometry_window

from src.flair_zonal_detection.config import (
    load_config,
    validate_config,
    summarize_config,
)
from src.flair_hub.utils.messaging import Logger
from src.flair_zonal_detection.dataset import MultiModalSlicedDataset
from src.flair_zonal_detection.postprocess import (
    convert,
    create_polygon_from_bounds,
    convert_to_cog
)
from src.flair_zonal_detection.model_utils import (
    build_inference_model,
    compute_patch_sizes
)
from src.flair_zonal_detection.slicing import generate_patches_from_reference


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
    summarize_config(config)

    config['device'] = torch.device("cuda" if config.get("use_gpu", torch.cuda.is_available()) else "cpu")
    config['output_type'] = config.get("output_type", "argmax")
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
    Initialize output raster files per task.
    """
    output_files = {}
    temp_paths = {}
    output_type = config['output_type']

    for task in config['tasks']:
        if not task['active']:
            continue

        num_classes = len(task['class_names'])
        suffix = 'argmax' if output_type == 'argmax' else 'class-prob'
        out_path = os.path.join(
            config['output_path'],
            f"{config['output_name']}_{task['name']}_{suffix}.tif"
        )
        profile = ref_img.profile.copy()
        profile.update({
            "count": num_classes if output_type == "class_prob" else 1,
            "dtype": "uint8",
            "compress": "lzw"
        })

        output_files[task['name']] = rasterio.open(out_path, 'w', **profile)
        temp_paths[task['name']] = out_path

    return output_files, temp_paths


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
    """
    device = config['device']
    margin = config['margin']
    tile_size = config['img_pixels_detection']
    output_type = config['output_type']

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
                logit_patch = logits[i, :, margin:tile_size - margin, margin:tile_size - margin]
                prediction = convert(logit_patch, output_type)

                patch_bounds = create_polygon_from_bounds(
                    row['left'], row['right'], row['bottom'], row['top']
                )
                window = geometry_window(
                    ref_img, [patch_bounds], pixel_precision=6
                ).round_offsets(op='ceil', pixel_precision=4)

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

        ref_img = rasterio.open(config['modalities'][config['reference_modality_for_slicing']]['input_img_path'])
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