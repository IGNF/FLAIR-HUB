import os
import numpy as np
import rasterio
import geopandas as gpd

from shapely.geometry import box


def create_box_from_bounds(x_min, x_max, y_min, y_max):
    return box(x_min, y_max, x_max, y_min)


def generate_patches_from_reference(config):
    ref_mod = config['reference_modality_for_slicing']
    img_path = config['modalities'][ref_mod]['input_img_path']
    patch_size = config['img_pixels_detection']
    margin = config['margin']
    output_path = config['output_path']
    output_name = config['output_name']
    write_dataframe = config.get('write_dataframe', False)

    with rasterio.open(img_path) as src:
        profile = src.profile
        left_overall, bottom_overall, right_overall, top_overall = src.bounds
        resolution = abs(round(src.res[0], 5)), abs(round(src.res[1], 5))

    geo_output_size = [patch_size * resolution[0], patch_size * resolution[1]]
    geo_margin = [margin * resolution[0], margin * resolution[1]]
    geo_step = [geo_output_size[0] - (2 * geo_margin[0]), geo_output_size[1] - (2 * geo_margin[1])]

    min_x, min_y = left_overall, bottom_overall
    max_x, max_y = right_overall, top_overall

    tmp_list = []
    existing_patches = set()

    for x_coord in np.arange(min_x - geo_margin[0], max_x + geo_margin[0], geo_step[0]):
        for y_coord in np.arange(min_y - geo_margin[1], max_y + geo_margin[1], geo_step[1]):

            if x_coord + geo_output_size[0] > max_x + geo_margin[0]:
                x_coord = max_x + geo_margin[0] - geo_output_size[0]
            if y_coord + geo_output_size[1] > max_y + geo_margin[1]:
                y_coord = max_y + geo_margin[1] - geo_output_size[1]

            left = x_coord + geo_margin[0]
            right = x_coord + geo_output_size[0] - geo_margin[0]
            bottom = y_coord + geo_margin[1]
            top = y_coord + geo_output_size[1] - geo_margin[1]

            right = min(right, max_x)
            top = min(top, max_y)

            col = int((x_coord - min_x) // resolution[0]) + 1
            row = int((y_coord - min_y) // resolution[1]) + 1

            new_patch = (round(left, 6), round(bottom, 6), round(right, 6), round(top, 6))
            if new_patch not in existing_patches:
                existing_patches.add(new_patch)
                tmp_list.append({
                    "id": str(f"{1}-{row}-{col}"),
                    "output_id": output_name,
                    "job_done": 0,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top,
                    "left_o": left_overall,
                    "bottom_o": bottom_overall,
                    "right_o": right_overall,
                    "top_o": top_overall,
                    "geometry": create_box_from_bounds(x_coord, x_coord + geo_output_size[0], y_coord, y_coord + geo_output_size[1]),
                })

    gdf_output = gpd.GeoDataFrame(tmp_list, crs=profile['crs'], geometry="geometry")

    if write_dataframe:
        gdf_output.to_file(os.path.join(output_path, output_name.split('.tif')[0]+'_slicing_job.gpkg'), driver='GPKG')

    return gdf_output
