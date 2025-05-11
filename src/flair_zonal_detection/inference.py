import os
import sys
import torch
import rasterio
import traceback
import datetime

from tqdm import tqdm
from torch.utils.data import DataLoader
from rasterio.features import geometry_window

from src.flair_zonal_detection.config import load_config, validate_config, summarize_config, Logger
from src.flair_zonal_detection.dataset import MultiModalSlicedDataset
from src.flair_zonal_detection.postprocess import convert, create_polygon_from_bounds, convert_to_cog
from src.flair_zonal_detection.model_utils import build_inference_model, compute_patch_sizes 
from src.flair_zonal_detection.slicing import generate_patches_from_reference




def prep_config(config_path):
    config = load_config(config_path)

    log_filename = os.path.join(config['output_path'], f"{config['output_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Logger(filename=log_filename)
    sys.stderr = sys.stdout
    print(f"\n[LOGGER] Writing logs to: {log_filename}")

    validate_config(config)
    summarize_config(config)

    config['device'] = torch.device("cuda" if config.get("use_gpu", torch.cuda.is_available()) else "cpu")
    config['output_type'] = config.get("output_type", "argmax")
    return config



def prep_dataset(config, tiles_gdf, patch_sizes):
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



def init_outputs(config, ref_img):
    output_files = {}
    temp_paths = {}
    output_type = config['output_type']

    for task in config['tasks']:
        if not task['active']:
            continue

        num_classes = len(task['class_names'])
        suffix = 'argmax' if output_type == 'argmax' else 'class-prob'
        out_path = os.path.join(config['output_path'], f"{config['output_name']}_{task['name']}_{suffix}.tif")
        profile = ref_img.profile.copy()
        count = num_classes if output_type == "class_prob" else 1
        profile.update({"count": count, "dtype": "uint8", "compress": "lzw"})

        dst = rasterio.open(out_path, 'w', **profile)
        output_files[task['name']] = dst
        temp_paths[task['name']] = out_path

    return output_files, temp_paths



def inference_and_write(model, dataloader, tiles_gdf, config, output_files, ref_img):
    device = config['device']
    margin = config['margin']
    tile_size = config['img_pixels_detection']
    output_type = config['output_type']

    print("\n[ ] Starting inference and writing raster tiles...\n")

    for batch in tqdm(dataloader):
        # Prepare model input
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
            logits = logits.cpu().numpy()  # B, C, H, W

            for i, idx in enumerate(indices):
                row = rows.iloc[i]
                logit_patch = logits[i, :, margin:tile_size - margin, margin:tile_size - margin]
                prediction = convert(logit_patch, output_type)

                patch_bounds = create_polygon_from_bounds(
                    row['left'], row['right'], row['bottom'], row['top']
                )
                window = geometry_window(
                    ref_img, [patch_bounds],
                    pixel_precision=6
                ).round_offsets(op='ceil', pixel_precision=4)

                if output_type == "argmax":
                    output_files[task_name].write(prediction[0], 1, window=window)
                else:
                    for c in range(prediction.shape[0]):
                        output_files[task_name].write(prediction[c], c + 1, window=window)
                        
    # Close raster files
    for dst in output_files.values():
        dst.close()




def postpro_outputs(temp_paths, config):
    if config.get("cog_conversion", False):
        for task_name, temp_path in temp_paths.items():
            cog_path = temp_path.replace(".tif", "_COG.tif")
            convert_to_cog(temp_path, cog_path)
            print(f"\n[✓] Converted to COG.")




def run_inference(config_path):
    try:
        config = prep_config(config_path)
        tiles_gdf = generate_patches_from_reference(config)
        print(f"\n[✓] Slicing input image into {len(tiles_gdf)} squares.")

        patch_sizes = compute_patch_sizes(config)
        model = build_inference_model(config, patch_sizes).to(config['device'])
        print(f"\n[✓] Loaded model and checkpoint.")

        dataset = prep_dataset(config, tiles_gdf, patch_sizes)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_worker'])

        ref_img = rasterio.open(config['modalities'][config['reference_modality_for_slicing']]['input_img_path'])
        output_files, temp_paths = init_outputs(config, ref_img)

        inference_and_write(model, dataloader, tiles_gdf, config, output_files, ref_img)
        postpro_outputs(temp_paths, config)

        print(f"\n[✓] Inference complete. Rasters written to {temp_paths}\n")

    except Exception as e:
        print("\n[✗] Inference failed with an error:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
    finally:
        # Ensure stdout is restored even if error occurs
        sys.stdout = sys.__stdout__
