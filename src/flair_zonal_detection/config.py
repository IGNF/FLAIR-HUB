"""Configuration loading for zonal inference.

Zonal runs contain only deployment inputs. Architecture, tasks, band order and
normalization are loaded from the training run's resolved_config.yaml.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import yaml

from flair_hub.utils.config_schema import load_and_compile_public_config


def _relative_to(config_file: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_file.parent / path
    return str(path.resolve())


def load_config(path: str) -> dict:
    """Load a zonal config and merge its immutable training configuration."""
    config_file = Path(path).resolve()
    with config_file.open("r", encoding="utf-8") as stream:
        zonal = yaml.safe_load(stream) or {}

    artifacts = zonal.get("training_artifacts", {})
    if "resolved_config" not in artifacts or "checkpoint" not in artifacts:
        raise ValueError(
            "training_artifacts.resolved_config and training_artifacts.checkpoint are required"
        )
    resolved_path = _relative_to(config_file, artifacts["resolved_config"])
    checkpoint_path = _relative_to(config_file, artifacts["checkpoint"])
    with open(resolved_path, "r", encoding="utf-8") as stream:
        public_training = yaml.safe_load(stream) or {}
    runtime = load_and_compile_public_config(public_training)
    runtime["paths"]["ckpt_model_path"] = checkpoint_path

    inference = zonal.get("inference", {})
    output = zonal.get("output", {})
    declared_inputs = zonal.get("inputs", {})
    runtime_modalities = runtime["modalities"]
    modalities = {"inputs": deepcopy(runtime_modalities["inputs"])}
    for modality, active in modalities["inputs"].items():
        if not active:
            continue
        if modality not in declared_inputs:
            raise ValueError(f"Missing zonal raster declaration for active modality {modality}")
        source = deepcopy(declared_inputs[modality])
        unknown = set(source) - {'raster', 'dates_file', 'cloud_mask_raster'}
        if unknown:
            raise ValueError(f'{modality}: deployment cannot override training options: {sorted(unknown)}')
        if "raster" not in source:
            raise ValueError(f"inputs.{modality}.raster is required")
        modality_cfg = {
            "input_img_path": _relative_to(config_file, source.pop("raster")),
            "channels": deepcopy(runtime_modalities["read_channels"].get(modality, [])),
            "loaded_band_order": deepcopy(runtime_modalities["read_channels"].get(modality, [])),
            "input_band_order": deepcopy(runtime_modalities["inputs_channels"].get(modality, [])),
            "normalization": {
                "type": runtime_modalities["normalization"].get("types", {}).get(
                    modality, runtime_modalities["normalization"]["norm_type"]
                ),
                "mean": deepcopy(runtime_modalities["normalization"].get(f"{modality}_means", [])),
                "std": deepcopy(runtime_modalities["normalization"].get(f"{modality}_stds", [])),
                "scale": runtime_modalities["normalization"].get(f"{modality}_scale", 1.0),
                "offset": runtime_modalities["normalization"].get(f"{modality}_offset", 0.0),
            },
            **source,
        }
        preprocess = runtime_modalities["pre_processings"]
        if modality == "DEM_ELEV":
            modality_cfg["derive_elevation"] = preprocess["calc_elevation"]
            modality_cfg["stack_with_dsm"] = preprocess["calc_elevation_stack_dsm"]
        elif modality == "SENTINEL2_TS":
            modality_cfg["filter_clouds"] = preprocess["filter_sentinel2"]
            modality_cfg['max_cloud'] = preprocess['filter_sentinel2_max_cloud']
            modality_cfg['max_snow'] = preprocess['filter_sentinel2_max_snow']
            modality_cfg['maximum_covered_fraction'] = preprocess['filter_sentinel2_max_frac_cover']
            modality_cfg["temporal_average"] = preprocess["temporal_average_sentinel2"]
        elif modality.startswith("SENTINEL1"):
            modality_cfg["temporal_average"] = preprocess["temporal_average_sentinel1"]
        for path_key in ("dates_file", "cloud_mask_raster"):
            if modality_cfg.get(path_key):
                modality_cfg[path_key] = _relative_to(config_file, modality_cfg[path_key])
        modality_cfg["dates_txt"] = modality_cfg.get("dates_file")
        modality_cfg["filter_clouds_img_path"] = modality_cfg.get("cloud_mask_raster")
        modalities[modality] = modality_cfg

    output_path = _relative_to(config_file, output.get("folder", "."))
    merged = {
        "output_path": output_path,
        "output_name": output.get("name", "flair_prediction"),
        "write_dataframe": bool(output.get("write_slicing_grid", False)),
        "output_type": output.get("type", "argmax"),
        "cog_conversion": bool(output.get("cloud_optimized_geotiff", True)),
        "model_weights": checkpoint_path,
        "use_gpu": bool(inference.get("use_gpu", True)),
        "batch_size": int(inference.get("batch_size", 1)),
        "num_worker": int(inference.get("num_workers", 0)),
        "img_pixels_detection": int(inference.get("patch_size", 2048)),
        "margin": int(inference.get("margin", 512)),
        "output_px_meters": float(inference.get("output_resolution", 0.2)),
        "precision": str(inference.get("precision", "auto")),
        "modalities": modalities,
        "labels": runtime["labels"],
        "labels_configs": runtime["labels_configs"],
        "multitemp_model_ref_date": runtime["models"]["multitemp_model"]["ref_date"],
        "runtime_model_config": runtime,
        "resolved_config_path": resolved_path,
    }
    return merged


def validate_config(config: dict) -> None:
    """Fail before slicing or model construction if deployment files are invalid."""
    if not os.path.isfile(config["model_weights"]):
        raise FileNotFoundError(f"Model checkpoint not found: {config['model_weights']}")
    if config['batch_size'] < 1 or config['num_worker'] < 0 or config['margin'] < 0 or config['output_px_meters'] <= 0:
        raise ValueError('Invalid deployment batch size, workers, margin or resolution')
    if config["img_pixels_detection"] <= 2 * config["margin"]:
        raise ValueError("inference.patch_size must be greater than twice inference.margin")
    if config["output_type"] not in {"argmax", "class_prob"}:
        raise ValueError("output.type must be argmax or class_prob")
    if config.get("precision", "auto") not in {"auto", "32-true", "16-mixed", "bf16-mixed"}:
        raise ValueError("inference.precision must be auto, 32-true, 16-mixed or bf16-mixed")
    for modality, active in config["modalities"]["inputs"].items():
        if active and not os.path.isfile(config["modalities"][modality]["input_img_path"]):
            raise FileNotFoundError(
                f"Input raster not found for {modality}: "
                f"{config['modalities'][modality]['input_img_path']}"
            )
        if active and modality.endswith('_TS'):
            source = config['modalities'][modality]
            if not source.get('dates_file') or not os.path.isfile(source['dates_file']):
                raise ValueError(f'{modality}: dates_file is required')
            if source.get('filter_clouds') and (not source.get('cloud_mask_raster') or
                                                not os.path.isfile(source['cloud_mask_raster'])):
                raise ValueError(f'{modality}: cloud_mask_raster is required')
    os.makedirs(config["output_path"], exist_ok=True)


def config_recap_1(config: dict) -> None:
    used_mods = ", ".join(m for m, active in config["modalities"]["inputs"].items() if active)
    active_tasks = ", ".join(config["labels"])
    device = str(config.get("device", "cuda" if config.get("use_gpu", False) else "cpu"))
    output_type = config["output_type"]
    output_display = f"{output_type} (COG)" if config.get("cog_conversion", False) else output_type
    print(
        "\n##############################################\n"
        "FLAIR-HUB ZONE DETECTION\n"
        "##############################################\n"
        f"|→ Resolved training config : {config['resolved_config_path']}\n"
        f"|→ Checkpoint path          : {config['model_weights']}\n"
        f"|→ Output                   : {config['output_path']} / {config['output_name']}\n"
        f"|→ Modalities               : {used_mods}\n"
        f"|→ Tasks                    : {active_tasks}\n"
        f"|→ Output type              : {output_display}\n"
        f"|→ Precision                : {resolve_inference_precision(config)} "
        f"(requested: {config.get('precision', 'auto')})\n"
        f"|→ Device / batch / workers : {device} / {config['batch_size']} / {config['num_worker']}"
    )


def resolve_inference_precision(config: dict) -> str:
    """Inherit training precision unless deployment explicitly overrides it."""
    precision = config.get("precision", "auto")
    if precision == "auto":
        precision = str(config["runtime_model_config"]["hardware"].get("precision", "32-true"))
    if precision not in {"32-true", "16-mixed", "bf16-mixed"}:
        raise ValueError(f"Unsupported inference precision: {precision}; set inference.precision explicitly")
    if str(config.get("device", "cpu")).startswith("cpu"):
        return "32-true"
    return precision


def config_recap_2(config: dict) -> None:
    def fmt(value, precision=5):
        return f"{value:.{precision}f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)

    shape = config.get("image_shape_px", {})
    resolution = config["reference_resolution"]
    if shape:
        print(f"|→ Image size               : {shape['height']} × {shape['width']} px")
    print(f"|→ Reference resolution     : {fmt(resolution)} m/px")
    print(f"|→ Output resolution        : {fmt(config['output_px_meters'])} m/px")
    print(
        f"|→ Patch / margin           : {config['img_pixels_detection']} / {config['margin']} px "
        f"({fmt(config['img_pixels_detection'] * resolution)} / {fmt(config['margin'] * resolution)} m)"
    )
    print("|→ Modality resolutions     :")
    for modality, value in config["modality_resolutions"].items():
        print(f"   - {modality:20}: {fmt(value)} m/px")
    print("=" * 46 + "\n")
