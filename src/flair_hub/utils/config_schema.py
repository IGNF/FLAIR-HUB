"""Public configuration schema and compiler for FLAIR-HUB V2.

The YAML files expose stable, explicit concepts to users.  The compiler keeps
the existing model/training implementation isolated behind its historical
runtime dictionary, so configuration cleanup does not change checkpoint
semantics.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Dict


MONOTEMPORAL_MODALITIES = {
    "AERIAL_RGBI",
    "AERIAL-RLT_PAN",
    "DEM_ELEV",
    "SPOT_RGBI",
}
TEMPORAL_MODALITIES = {
    "SENTINEL2_TS",
    "SENTINEL1-ASC_TS",
    "SENTINEL1-DESC_TS",
}


def _require(mapping: dict, key: str, location: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing configuration key: {location}.{key}")
    return mapping[key]


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{location} must be a strictly positive integer")
    return int(value)


def _probability(value: Any, location: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{location} must be between 0 and 1")
    return value


def _validate_band_order(modality: str, cfg: dict) -> None:
    if modality == "DEM_ELEV":
        cfg.setdefault("input_band_order", [1, 2])
    order = _require(cfg, "input_band_order", f"data.modalities.{modality}")
    if not isinstance(order, list) or not order:
        raise ValueError(
            f"data.modalities.{modality}.input_band_order must be a non-empty list"
        )
    if any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in order):
        raise ValueError(
            f"data.modalities.{modality}.input_band_order must contain positive band numbers"
        )
    if len(set(int(v) for v in order)) != len(order):
        raise ValueError(
            f"data.modalities.{modality}.input_band_order cannot contain duplicates"
        )


def _normalization_channel_count(modality: str, cfg: dict, preprocessing: dict) -> int:
    if modality != "DEM_ELEV":
        return len(cfg["input_band_order"])
    elevation = preprocessing.get("elevation", {})
    if elevation.get("derive_elevation", False):
        return 2 if elevation.get("stack_with_dsm", False) else 1
    return len(cfg.get("input_band_order", [1, 2]))


def resolve_public_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fill defaults in the public V2 configuration."""
    cfg = deepcopy(config)
    cfg.setdefault("logging", {})
    cfg.setdefault("checkpoints", {})
    required_sections = (
        "experiment",
        "execution",
        "training",
        "hardware",
        "checkpoints",
        "logging",
        "model",
        "data",
        "segmentation_tasks",
    )
    for section in required_sections:
        _require(cfg, section, "configuration")

    experiment = cfg["experiment"]
    for key in (
        "output_folder",
        "name",
        "train_csv",
        "validation_csv",
        "test_csv",
        "metadata_folder",
    ):
        _require(experiment, key, "experiment")

    training = cfg["training"]
    duration = _require(training, "duration", "training")
    unit = str(_require(duration, "unit", "training.duration")).lower()
    if unit not in {"steps", "epochs"}:
        raise ValueError("training.duration.unit must be 'steps' or 'epochs'")
    duration["unit"] = unit
    duration["total"] = _positive_int(
        _require(duration, "total", "training.duration"),
        "training.duration.total",
    )
    validation = _require(training, "validation", "training")
    validation["every"] = _positive_int(
        _require(validation, "every", "training.validation"),
        "training.validation.every",
    )
    if "batch" in training:
        raise ValueError(
            "training.batch has been replaced by the simpler "
            "training.batch_size setting"
        )
    training["batch_size"] = _positive_int(
        _require(training, "batch_size", "training"),
        "training.batch_size",
    )

    optimizer = _require(training, "optimizer", "training")
    optimizer_name = str(_require(optimizer, "name", "training.optimizer")).lower()
    if optimizer_name not in {"adamw", "adam", "sgd"}:
        raise ValueError(
            "training.optimizer.name must be adamw, adam or sgd"
        )
    optimizer["name"] = optimizer_name
    if "paramwise_weight_decay" in optimizer:
        raise ValueError(
            "training.optimizer.paramwise_weight_decay is now managed "
            "internally and must be removed"
        )

    scheduler = _require(training, "scheduler", "training")
    scheduler_name = scheduler.get("name")
    if scheduler_name is None or str(scheduler_name).lower() == "none":
        scheduler["name"] = None
    else:
        scheduler_name = str(scheduler_name).lower()
        if scheduler_name not in {
            "cosine",
            "one_cycle_lr",
            "reduce_on_plateau",
            "cycle_then_plateau",
        }:
            raise ValueError(
                "training.scheduler.name must be cosine, one_cycle_lr, "
                "reduce_on_plateau, cycle_then_plateau or none"
            )
        scheduler["name"] = scheduler_name

    hardware = cfg["hardware"]
    for key in ("num_nodes", "gpus_per_node"):
        hardware[key] = _positive_int(
            _require(hardware, key, "hardware"), f"hardware.{key}"
        )
    workers = hardware.setdefault("num_workers", 0)
    if isinstance(workers, bool) or int(workers) != workers or workers < 0:
        raise ValueError("hardware.num_workers must be a non-negative integer")
    hardware["num_workers"] = int(workers)
    hardware.setdefault("pin_memory", "auto")
    if hardware["pin_memory"] not in (True, False, "auto"):
        raise ValueError("hardware.pin_memory must be true, false or auto")
    hardware["accelerator"] = str(
        _require(hardware, "accelerator", "hardware")
    ).lower()
    if hardware["accelerator"] not in {"gpu", "cpu", "auto"}:
        raise ValueError("hardware.accelerator must be gpu, cpu or auto")
    hardware["strategy"] = str(hardware.get(
        "strategy", "auto" if hardware["num_nodes"] * hardware["gpus_per_node"] == 1
        else "ddp_find_unused_parameters_true")).lower()
    supported_strategies = {
        "auto",
        "ddp",
        "ddp_find_unused_parameters_true",
    }
    if hardware["strategy"] not in supported_strategies:
        raise ValueError(
            "hardware.strategy must be auto, ddp or "
            "ddp_find_unused_parameters_true"
        )

    public_logging = cfg["logging"]
    removed_progress_keys = {
        "progress_bar",
        "progress_report_every_steps",
    } & set(public_logging)
    if removed_progress_keys:
        raise ValueError(
            "logging.progress_bar/progress_report_every_steps have been "
            "replaced by the single logging.progress setting"
        )
    public_logging["progress"] = bool(public_logging.get("progress", True))
    for key, default in {"checkpoint_messages": True, "verbose_config": False,
                         "copy_input_csv": True}.items():
        public_logging.setdefault(key, default)
    checkpoints = cfg["checkpoints"]
    checkpoints.setdefault("initialize_from", None)
    checkpoints.setdefault("resume_from", None)
    checkpoints.setdefault("reset_task_heads", False)
    for section, defaults in {
        "best": {"monitor": "val_miou", "mode": "max", "weights_only": False, "also_save_last": False},
        "recovery": {"enabled": False, "every_steps": 2000, "directory": "recovery_checkpoints"},
        "early_stopping": {"enabled": False, "patience": 20},
    }.items():
        target = checkpoints.setdefault(section, {})
        for key, value in defaults.items():
            target.setdefault(key, value)

    monotemporal = _require(cfg["model"], "monotemporal", "model")
    encoder = _require(monotemporal, "encoder", "model.monotemporal")
    decoder = _require(monotemporal, "decoder", "model.monotemporal")
    if "new_channels_init_mode" in encoder:
        raise ValueError(
            "model.monotemporal.encoder.new_channels_init_mode was unused "
            "and has been removed"
        )
    fusion = cfg["model"].setdefault("fusion", {})
    fusion["feature_source"] = str(
        fusion.get("feature_source", "encoder")
    ).lower()
    if fusion["feature_source"] not in {"encoder", "decoder"}:
        raise ValueError("model.fusion.feature_source must be encoder or decoder")
    fusion["method"] = str(
        fusion.get("method", "gated_residual")
    ).lower()
    if fusion["method"] not in {
        "aligned_concat",
        "gated_residual",
        "window_cross_attention",
    }:
        raise ValueError(
            "model.fusion.method must be aligned_concat, gated_residual or "
            "window_cross_attention"
        )
    levels = fusion.get("levels", [1, 2, 3])
    if (
        not isinstance(levels, list)
        or not levels
        or any(isinstance(level, bool) or int(level) != level for level in levels)
    ):
        raise ValueError(
            "model.fusion.levels must be a non-empty list of integer Swin "
            "stage indices"
        )
    fusion["levels"] = [int(level) for level in levels]
    if len(set(fusion["levels"])) != len(fusion["levels"]):
        raise ValueError("model.fusion.levels cannot contain duplicates")
    invalid_levels = [level for level in fusion["levels"] if level not in range(4)]
    if invalid_levels:
        raise ValueError(
            "model.fusion.levels must use indices 0, 1, 2 or 3; "
            f"got {invalid_levels}"
        )
    fusion["gate_init"] = float(fusion.get("gate_init", -2.0))
    fusion["attention_channels"] = _positive_int(
        fusion.get("attention_channels", 128),
        "model.fusion.attention_channels",
    )
    fusion["attention_heads"] = _positive_int(
        fusion.get("attention_heads", 4),
        "model.fusion.attention_heads",
    )
    fusion["window_size"] = _positive_int(
        fusion.get("window_size", 8),
        "model.fusion.window_size",
    )
    if fusion["attention_channels"] % fusion["attention_heads"] != 0:
        raise ValueError(
            "model.fusion.attention_channels must be divisible by "
            "model.fusion.attention_heads"
        )
    _require(encoder, "name", "model.monotemporal.encoder")
    decoder_name = str(_require(decoder, "name", "model.monotemporal.decoder")).lower()
    decoder["name"] = decoder_name
    decoder["provider"] = str(decoder.get("provider", "smp")).lower()
    if "parameters" in decoder:
        raise ValueError(
            "model.monotemporal.decoder.parameters has been renamed to "
            "model.monotemporal.decoder.upernet_flair_hub_params"
        )
    if decoder_name != "upernet" and decoder["provider"] != "smp":
        raise ValueError(
            "Only UPerNet can use a decoder provider other than 'smp'"
        )
    if decoder["provider"] not in {"smp", "flair_hub"}:
        raise ValueError("model.monotemporal.decoder.provider must be 'smp' or 'flair_hub'")
    if "flair_hub_parameters" in decoder:
        legacy = decoder.pop("flair_hub_parameters")
        if "upernet_flair_hub_params" in decoder and decoder["upernet_flair_hub_params"] != legacy:
            raise ValueError("Conflicting UPerNet parameter blocks")
        decoder.setdefault("upernet_flair_hub_params", legacy)
    flair_hub_parameters = decoder.get("upernet_flair_hub_params")
    if (
        flair_hub_parameters is not None
        and not isinstance(flair_hub_parameters, dict)
    ):
        raise ValueError(
            "model.monotemporal.decoder.upernet_flair_hub_params must be a mapping"
        )
    if decoder["provider"] == "smp" and "upernet_flair_hub_params" in decoder:
        raise ValueError(
            "model.monotemporal.decoder.upernet_flair_hub_params is valid only "
            "for the FLAIR-HUB UPerNet implementation; remove this block when "
            "model.monotemporal.decoder.provider is 'smp'"
        )
    if decoder["provider"] == "flair_hub" and not flair_hub_parameters:
        raise ValueError(
            "model.monotemporal.decoder.upernet_flair_hub_params is required "
            "when model.monotemporal.decoder.provider is 'flair_hub'"
        )
    if decoder["provider"] == "smp":
        encoder["drop_path_rate"] = None
    supported_head_norms = {"sync_batchnorm", "batchnorm", "groupnorm"}
    decoder_norm = (flair_hub_parameters or {}).get("norm", "sync_batchnorm")
    if decoder_norm not in supported_head_norms:
        raise ValueError(
            "model.monotemporal.decoder.upernet_flair_hub_params.norm must be "
            "sync_batchnorm, batchnorm or groupnorm"
        )
    auxiliary_head = monotemporal.get("auxiliary_head", {})
    if "feature_index" in auxiliary_head:
        raise ValueError(
            "model.monotemporal.auxiliary_head.feature_index is no longer "
            "configurable; the penultimate encoder stage is selected automatically"
        )
    auxiliary_norm = auxiliary_head.get("norm", "sync_batchnorm")
    if auxiliary_norm not in supported_head_norms:
        raise ValueError(
            "model.monotemporal.auxiliary_head.norm must be "
            "sync_batchnorm, batchnorm or groupnorm"
        )

    multitemporal = _require(cfg["model"], "multitemporal", "model")
    hidden_channels = multitemporal.get("head_hidden_channels")
    if hidden_channels is not None:
        if not isinstance(hidden_channels, list) or not hidden_channels:
            raise ValueError("model.multitemporal.head_hidden_channels must be a non-empty list")
        multitemporal["head_hidden_channels"] = [
            _positive_int(value, "model.multitemporal.head_hidden_channels")
            for value in hidden_channels
        ]
        if "out_conv" in multitemporal:
            raise ValueError("Use head_hidden_channels or legacy out_conv, not both")
    if multitemporal.get("agg_mode", "att_group") not in {
        "att_group", "att_mean", "mean"
    }:
        raise ValueError(
            "model.multitemporal.agg_mode must be att_group, att_mean or mean"
        )
    if multitemporal.get("encoder_norm", "group") not in {
        "group", "batch", "instance"
    }:
        raise ValueError(
            "model.multitemporal.encoder_norm must be group, batch or instance"
        )
    if multitemporal.get("padding_mode", "reflect") not in {
        "zeros", "reflect", "replicate", "circular"
    }:
        raise ValueError(
            "model.multitemporal.padding_mode must be zeros, reflect, "
            "replicate or circular"
        )
    data = cfg["data"]
    modalities = _require(data, "modalities", "data")
    for modality in sorted(MONOTEMPORAL_MODALITIES | TEMPORAL_MODALITIES):
        modalities.setdefault(modality, {"enabled": False})
    preprocessing = data.setdefault("preprocessing", {})
    active_modalities = []
    for modality, modality_cfg in modalities.items():
        modality_cfg["enabled"] = bool(modality_cfg.get("enabled", False))
        if not modality_cfg["enabled"]:
            continue
        active_modalities.append(modality)
        if modality not in MONOTEMPORAL_MODALITIES | TEMPORAL_MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}")
        _validate_band_order(modality, modality_cfg)
        if modality == "DEM_ELEV" and preprocessing.get("elevation", {}).get("derive_elevation", False):
            if len(modality_cfg["input_band_order"]) != 2:
                raise ValueError("DEM elevation derivation requires [DSM band, DTM band]")
        modality_cfg["dropout_probability"] = _probability(
            modality_cfg.get("dropout_probability", 0.0),
            f"data.modalities.{modality}.dropout_probability",
        )
        auxiliary = modality_cfg.setdefault("auxiliary_loss", {})
        auxiliary["enabled"] = bool(auxiliary.get("enabled", False))
        auxiliary["weight"] = float(auxiliary.get("weight", 0.4))
        normalization = modality_cfg.setdefault("normalization", {"type": "without"})
        normalization["type"] = str(normalization.get("type", "without")).lower()
        if normalization["type"] not in {"custom", "scaling", "without"}:
            raise ValueError(
                f"data.modalities.{modality}.normalization.type is invalid"
            )
        if normalization["type"] == "scaling":
            normalization.setdefault("scale", 1.0 / 255.0 if modality in {
                "AERIAL_RGBI", "AERIAL-RLT_PAN"} else None)
            if normalization["scale"] is None:
                raise ValueError(f"{modality}: scaling requires an explicit normalization.scale")
            normalization["scale"] = float(normalization["scale"])
            normalization["offset"] = float(normalization.get("offset", 0.0))
        if normalization["type"] == "custom":
            count = _normalization_channel_count(modality, modality_cfg, preprocessing)
            mean = _require(
                normalization, "mean", f"data.modalities.{modality}.normalization"
            )
            std = _require(
                normalization, "std", f"data.modalities.{modality}.normalization"
            )
            if len(mean) != count or len(std) != count:
                raise ValueError(
                    f"{modality}: normalization mean/std must have {count} values, "
                    "in exactly the same order as input_band_order"
                )
            if any(float(value) == 0 for value in std):
                raise ValueError(f"{modality}: normalization std values must be non-zero")
    if not active_modalities:
        raise ValueError("At least one data modality must be enabled")

    augmentation = data.setdefault("augmentation", {})
    augmentation["enabled"] = bool(augmentation.get("enabled", False))
    geometry = augmentation.setdefault("geometry", {})
    geometry.setdefault("scale_range", [1.0, 1.0])
    geometry.setdefault("category_max_ratio", 1.0)
    geometry.setdefault("horizontal_flip_probability", 0.0)
    geometry.setdefault("vertical_flip_probability", 0.0)
    geometry.setdefault("rotation_probability", 0.0)
    geometry.setdefault("rotation_degrees", [0.0, 0.0])
    if len(geometry["scale_range"]) != 2 or min(geometry["scale_range"]) <= 0:
        raise ValueError("data.augmentation.geometry.scale_range must contain two positive values")
    if "crop_size" in geometry:
        raise ValueError(
            "data.augmentation.geometry.crop_size is no longer configurable; "
            "the crop size is inferred from each patch before augmentation"
        )
    for key in (
        "horizontal_flip_probability",
        "vertical_flip_probability",
        "rotation_probability",
    ):
        geometry[key] = _probability(
            geometry[key], f"data.augmentation.geometry.{key}"
        )

    radiometry = augmentation.setdefault("radiometry", {})
    if not isinstance(radiometry, dict):
        raise ValueError("data.augmentation.radiometry must be a mapping")
    for modality, radiometry_cfg in radiometry.items():
        if modality not in modalities:
            raise ValueError(
                f"data.augmentation.radiometry references unknown modality {modality}"
            )
        if not isinstance(radiometry_cfg, dict):
            raise ValueError(
                f"data.augmentation.radiometry.{modality} must be a mapping"
            )
        radiometry_cfg["enabled"] = bool(radiometry_cfg.get("enabled", False))
        if not modalities[modality]["enabled"]:
            radiometry_cfg["enabled"] = False
            continue
        if not augmentation["enabled"] or not radiometry_cfg["enabled"]:
            continue
        band_order = modalities[modality].get("input_band_order") or []
        mode = radiometry_cfg.setdefault("mode",
            "hsv" if modality == "AERIAL_RGBI" and len(band_order) == 3 else "gain_offset")
        if mode not in {"hsv", "gain_offset"}:
            raise ValueError(f"{modality}: radiometry.mode must be hsv or gain_offset")
        if mode == "hsv" and len(band_order) != 3:
            raise ValueError(
                f"HSV radiometry for {modality} requires exactly three bands in "
                f"input_band_order; got {band_order}"
            )
        if mode == "gain_offset":
            radiometry_cfg.setdefault("gain_range", [0.9, 1.1])
            radiometry_cfg.setdefault("offset_range", [0.0, 0.0])
            for key in ("gain_range", "offset_range"):
                values = radiometry_cfg[key]
                if len(values) != 2 or values[0] > values[1]:
                    raise ValueError(f"{modality}: {key} must be an ordered pair")

    tasks = cfg["segmentation_tasks"]
    enabled_tasks = [name for name, task in tasks.items() if task.get("enabled", False)]
    if not enabled_tasks:
        raise ValueError("At least one segmentation task must be enabled")
    if "crop_reference_task" in geometry:
        legacy = geometry.pop("crop_reference_task")
        if "cropping_reference_task" in geometry and geometry["cropping_reference_task"] != legacy:
            raise ValueError("Conflicting cropping reference tasks")
        geometry.setdefault("cropping_reference_task", legacy)
    crop_reference = geometry.get("cropping_reference_task")
    if len(enabled_tasks) == 1:
        geometry["cropping_reference_task"] = enabled_tasks[0]
        crop_reference = enabled_tasks[0]
    if crop_reference is not None and crop_reference not in enabled_tasks:
        raise ValueError(
            "data.augmentation.geometry.cropping_reference_task must name an enabled segmentation task"
        )
    for task in tasks.values():
        task.pop("model_output", None)
    for task_name in enabled_tasks:
        task = tasks[task_name]
        _require(task, "classes", f"segmentation_tasks.{task_name}")
        task.setdefault("task_weight", 1.0)
        weights = task.setdefault("class_weights", {})
        weights["default"] = float(weights.get("default", 1.0))
        if weights["default"] < 0:
            raise ValueError(
                f"segmentation_tasks.{task_name}.class_weights.default "
                "must be non-negative"
            )
        exceptions = weights.setdefault("exceptions", {})
        if not isinstance(exceptions, dict):
            raise ValueError(
                f"segmentation_tasks.{task_name}.class_weights.exceptions "
                "must be a class_id: weight dictionary"
            )
        class_ids = sorted(int(value) for value in task["classes"])
        if (not class_ids or len(set(class_ids)) != len(class_ids)
                or any(code < 0 or code > 4294967295 for code in class_ids)
                or any(str(k) != str(int(k)) for k in task["classes"])):
            raise ValueError(f"{task_name}: classes must have unique non-negative uint32 integer codes")
        task["classes"] = {int(k): v for k, v in task["classes"].items()}
        legacy_ignored = task.pop("ignore_codes", [])
        normalized_exceptions = {
            int(class_id): float(weight)
            for class_id, weight in exceptions.items()
        }
        invalid_weight_ids = sorted(set(normalized_exceptions) - set(class_ids))
        if invalid_weight_ids:
            raise ValueError(
                f"segmentation_tasks.{task_name}.class_weights.exceptions "
                f"contains unknown class IDs {invalid_weight_ids}"
            )
        if any(weight < 0 for weight in normalized_exceptions.values()):
            raise ValueError(
                f"segmentation_tasks.{task_name}.class_weights.exceptions "
                "weights must be non-negative"
            )
        weights["exceptions"] = normalized_exceptions
        for code in legacy_ignored:
            if int(code) in task["classes"]:
                normalized_exceptions[int(code)] = 0.0
        globally_ignored_ids = {
            class_id
            for class_id in class_ids
            if normalized_exceptions.get(class_id, weights["default"]) == 0
        }

        per_modality = weights.setdefault("per_modality_exceptions", {})
        if not isinstance(per_modality, dict):
            raise ValueError(
                f"segmentation_tasks.{task_name}.class_weights."
                "per_modality_exceptions must map modality names to "
                "class_id: weight dictionaries"
            )
        normalized_per_modality = {}
        for modality, modality_exceptions in per_modality.items():
            if modality not in modalities:
                raise ValueError(
                    f"segmentation_tasks.{task_name}.class_weights."
                    f"per_modality_exceptions contains unknown modality {modality!r}"
                )
            if not isinstance(modality_exceptions, dict):
                raise ValueError(
                    f"segmentation_tasks.{task_name}.class_weights."
                    f"per_modality_exceptions.{modality} must be a "
                    "class_id: weight dictionary"
                )
            normalized_modality_exceptions = {
                int(class_id): float(weight)
                for class_id, weight in modality_exceptions.items()
            }
            invalid_modality_ids = sorted(
                set(normalized_modality_exceptions) - set(class_ids)
            )
            if invalid_modality_ids:
                raise ValueError(
                    f"segmentation_tasks.{task_name}.class_weights."
                    f"per_modality_exceptions.{modality} contains unknown "
                    f"class IDs {invalid_modality_ids}"
                )
            if any(
                weight < 0
                for weight in normalized_modality_exceptions.values()
            ):
                raise ValueError(
                    f"segmentation_tasks.{task_name}.class_weights."
                    f"per_modality_exceptions.{modality} weights must be "
                    "non-negative"
                )
            ineffective_ids = sorted(
                set(normalized_modality_exceptions) & globally_ignored_ids
            )
            if ineffective_ids:
                raise ValueError(
                    f"segmentation_tasks.{task_name}.class_weights."
                    f"per_modality_exceptions.{modality} cannot override "
                    f"globally ignored class IDs {ineffective_ids}; their "
                    "targets are converted to the internal ignore index before every loss"
                )
            normalized_per_modality[modality] = normalized_modality_exceptions
        weights["per_modality_exceptions"] = normalized_per_modality

        active_ids = [
            class_id for class_id in class_ids
            if weights["exceptions"].get(class_id, weights["default"]) > 0
        ]
        if not active_ids:
            raise ValueError(
                f"segmentation_tasks.{task_name} must contain at least one "
                "non-zero-weight class"
            )
        loss = task.setdefault("loss", {})
        ohem = loss.setdefault("ohem", {})
        ohem.setdefault("enabled", False)
        ohem.setdefault("threshold", 0.7)
        ohem.setdefault("minimum_pixels", 100000)

    cfg.pop("schema_version", None)
    for sensor in ("sentinel1", "sentinel2"):
        period = preprocessing.get(sensor, {}).get("temporal_average", False)
        if period not in (False, "monthly", "semi-monthly"):
            raise ValueError(f"{sensor}.temporal_average must be false, monthly or semi-monthly")
    return cfg


def compile_runtime_config(resolved: Dict[str, Any]) -> Dict[str, Any]:
    """Compile a resolved public configuration into the internal runtime form."""
    cfg = resolved
    experiment = cfg["experiment"]
    execution = cfg["execution"]
    training = cfg["training"]
    model = cfg["model"]
    modalities = cfg["data"]["modalities"]
    preprocessing = cfg["data"].get("preprocessing", {})
    duration = training["duration"]
    enabled_tasks = [
        name for name, task in cfg["segmentation_tasks"].items() if task["enabled"]
    ]
    total_classes = sum(
        sum(task["class_weights"]["exceptions"].get(code, task["class_weights"]["default"]) > 0
            for code in task["classes"])
        for name, task in cfg["segmentation_tasks"].items() if name in enabled_tasks
    )
    temporal_model = deepcopy(model["multitemporal"])
    if "head_hidden_channels" in temporal_model:
        temporal_model["out_conv"] = [*temporal_model.pop("head_hidden_channels"), total_classes]
    else:
        legacy = list(temporal_model.get("out_conv", [32]))
        temporal_model["out_conv"] = legacy if legacy[-1] == total_classes else [*legacy, total_classes]

    mono = model["monotemporal"]
    encoder = mono["encoder"]
    decoder = mono["decoder"]
    monotemp_runtime = {
        "arch": f"{encoder['name']}-{decoder['name']}",
        "encoder": {
            "weights": encoder.get("weights", "imagenet"),
            "drop_path_rate": encoder.get("drop_path_rate"),
        },
        "auxiliary_head": deepcopy(mono.get("auxiliary_head", {})),
    }
    if decoder["name"] == "upernet":
        monotemp_runtime["upernet_decoder"] = {
            "implementation": decoder["provider"],
            **deepcopy(decoder.get("upernet_flair_hub_params", {})),
        }

    norm_types = {
        name: modality_cfg.get("normalization", {}).get("type", "without")
        for name, modality_cfg in modalities.items()
    }
    inputs = {name: bool(value["enabled"]) for name, value in modalities.items()}
    input_orders = {
        name: deepcopy(value.get("input_band_order") or [])
        for name, value in modalities.items()
    }
    read_orders = {
        name: sorted(set(input_orders[name])) if name in MONOTEMPORAL_MODALITIES else input_orders[name]
        for name in modalities
    }
    normalization = {
        "norm_type": next(
            norm_types[name] for name, value in modalities.items() if value["enabled"]
        ),
        "types": norm_types,
    }
    for name, value in modalities.items():
        norm = value.get("normalization", {})
        normalization[f"{name}_means"] = deepcopy(norm.get("mean", []))
        normalization[f"{name}_stds"] = deepcopy(norm.get("std", []))
        normalization[f"{name}_scale"] = norm.get("scale", 1.0)
        normalization[f"{name}_offset"] = norm.get("offset", 0.0)

    elevation = preprocessing.get("elevation", {})
    sentinel2 = preprocessing.get("sentinel2", {})
    sentinel1 = preprocessing.get("sentinel1", {})
    modality_runtime = {
        "inputs": inputs,
        "inputs_channels": input_orders,
        "read_channels": read_orders,
        "aux_loss": {
            name: bool(value.get("auxiliary_loss", {}).get("enabled", False))
            for name, value in modalities.items()
        },
        "aux_loss_weight": {
            name: float(value.get("auxiliary_loss", {}).get("weight", 0.4))
            for name, value in modalities.items()
        },
        "modality_dropout": {
            name: float(value.get("dropout_probability", 0.0))
            for name, value in modalities.items()
        },
        "pre_processings": {
            "filter_sentinel2": bool(sentinel2.get("filter_clouds", False)),
            "filter_sentinel2_max_cloud": sentinel2.get("maximum_cloud", 100),
            "filter_sentinel2_max_snow": sentinel2.get("maximum_snow", 100),
            "filter_sentinel2_max_frac_cover": sentinel2.get("maximum_covered_fraction", 1.0),
            "temporal_average_sentinel2": sentinel2.get("temporal_average", False),
            "temporal_average_sentinel1": sentinel1.get("temporal_average", False),
            "calc_elevation": bool(elevation.get("derive_elevation", False)),
            "calc_elevation_stack_dsm": bool(elevation.get("stack_with_dsm", False)),
            "use_augmentation": bool(cfg["data"]["augmentation"]["enabled"]),
        },
        "normalization": normalization,
    }

    labels_configs = {}
    for name in enabled_tasks:
        task = cfg["segmentation_tasks"][name]
        weights = task["class_weights"]
        ohem = task["loss"]["ohem"]
        labels_configs[name] = {
            "task_weight": task["task_weight"],
            "label_channel_nomenclature": task.get("label_band", 1),
            "value_name": {int(k): v for k, v in task["classes"].items()},
            "nomenclature_version": task.get("nomenclature_version"),
            "value_weights": {
                "default": weights["default"],
                "default_exceptions": {
                    int(k): v for k, v in weights["exceptions"].items()
                },
                "per_modality_exceptions": deepcopy(weights["per_modality_exceptions"]),
            },
            "loss": {
                "ohem": {
                    "enabled": bool(ohem["enabled"]),
                    "thresh": float(ohem["threshold"]),
                    "min_kept": int(ohem["minimum_pixels"]),
                }
            },
        }

    optimizer = training["optimizer"]
    scheduler = training["scheduler"]
    checkpoints = cfg["checkpoints"]
    logging = cfg["logging"]
    if "resume" in checkpoints:
        raise ValueError(
            "checkpoints.resume.mode/path has been replaced by the single "
            "checkpoints.resume_from setting"
        )
    initialize_path = checkpoints.get("initialize_from")
    resume_from = checkpoints.get("resume_from")
    ensemble_members = checkpoints.get("ensemble")
    ensemble_paths = None
    ensemble_weights = None
    if ensemble_members is not None:
        if bool(execution.get("train", True)):
            raise ValueError(
                "checkpoints.ensemble is prediction-only; set execution.train to false"
            )
        if initialize_path is not None or resume_from is not None:
            raise ValueError(
                "checkpoints.ensemble cannot be combined with initialize_from or resume_from"
            )
        if not isinstance(ensemble_members, list) or len(ensemble_members) < 2:
            raise ValueError("checkpoints.ensemble requires at least two members")
        ensemble_paths = []
        ensemble_weights = []
        normalized_members = []
        for index, member in enumerate(ensemble_members):
            if not isinstance(member, dict):
                raise ValueError(
                    f"checkpoints.ensemble[{index}] must contain checkpoint and weight"
                )
            checkpoint_path = str(member.get("checkpoint", "")).strip()
            weight = float(member.get("weight", 1.0))
            if not checkpoint_path:
                raise ValueError(
                    f"checkpoints.ensemble[{index}].checkpoint must be a path"
                )
            if weight <= 0:
                raise ValueError(
                    f"checkpoints.ensemble[{index}].weight must be positive"
                )
            ensemble_paths.append(checkpoint_path)
            ensemble_weights.append(weight)
            normalized_members.append(
                {"checkpoint": checkpoint_path, "weight": weight}
            )
        total_weight = sum(ensemble_weights)
        ensemble_weights = [weight / total_weight for weight in ensemble_weights]
        checkpoints["ensemble"] = normalized_members
    if initialize_path is not None:
        initialize_path = str(initialize_path).strip()
        if not initialize_path:
            raise ValueError(
                "checkpoints.initialize_from must be null or a checkpoint path"
            )
        checkpoints["initialize_from"] = initialize_path
    if initialize_path is not None and resume_from is not None:
        raise ValueError(
            "checkpoints.initialize_from and checkpoints.resume_from are "
            "mutually exclusive"
        )

    if resume_from is None:
        runtime_resume_mode = False
        resume_path = None
    else:
        resume_value = str(resume_from).strip()
        if not resume_value:
            raise ValueError(
                "checkpoints.resume_from must be null, auto or a full "
                "Lightning checkpoint path"
            )
        if resume_value.lower() == "auto":
            checkpoints["resume_from"] = "auto"
            runtime_resume_mode = "auto"
            output_folder = str(experiment["output_folder"])
            path_class = PurePosixPath if output_folder.startswith("/") else Path
            resume_path = str(
                path_class(output_folder)
                / experiment["name"]
                / checkpoints.get("recovery", {}).get(
                    "directory", "recovery_checkpoints"
                )
                / "last.ckpt"
            )
        else:
            checkpoints["resume_from"] = resume_value
            runtime_resume_mode = True
            resume_path = resume_value

    best_mode = str(checkpoints.get("best", {}).get("mode", "max")).lower()
    if best_mode not in {"max", "min"}:
        raise ValueError("checkpoints.best.mode must be max or min")
    checkpoints.setdefault("best", {})["mode"] = best_mode
    runtime = {
        "models": {
            "monotemp_model": monotemp_runtime,
            "multitemp_model": temporal_model,
            "fusion": deepcopy(model["fusion"]),
        },
        "modalities": modality_runtime,
        "augmentation": deepcopy(cfg["data"]["augmentation"]),
        "labels": enabled_tasks,
        "labels_configs": labels_configs,
        "paths": {
            "out_folder": experiment["output_folder"],
            "out_model_name": experiment["name"],
            "train_csv": experiment["train_csv"],
            "val_csv": experiment["validation_csv"],
            "test_csv": experiment["test_csv"],
            "global_mtd_folder": experiment["metadata_folder"],
            "ckpt_model_path": resume_path or initialize_path,
            "ensemble_checkpoint_paths": ensemble_paths,
            "ensemble_weights": ensemble_weights,
        },
        "tasks": {
            "train": bool(execution.get("train", True)),
            "predict": bool(execution.get("predict", True)),
            "write_files": bool(execution.get("write_prediction_files", False)),
            "georeferencing_output": bool(execution.get("georeference_predictions", True)),
            "metrics_only": bool(execution.get("metrics_only", False)),
            "train_tasks": {
                "init_weights_only_from_ckpt": bool(initialize_path),
                "resume_training_from_ckpt": runtime_resume_mode,
            },
        },
        "hyperparams": {
            "num_epochs": duration["total"] if duration["unit"] == "epochs" else -1,
            "max_steps": duration["total"] if duration["unit"] == "steps" else None,
            "duration_unit": duration["unit"],
            "validation_every": training["validation"]["every"],
            "batch_size": training["batch_size"],
            "seed": training["seed"],
            "learning_rate": optimizer["learning_rate"],
            "optimizer": optimizer["name"],
            "optim_weight_decay": optimizer.get("weight_decay", 0.0),
            "optim_betas": optimizer.get("betas", [0.9, 0.999]),
            "scheduler": scheduler["name"],
            "warmup_steps": scheduler.get("warmup_steps", 0),
            "warmup_ratio": scheduler.get("warmup_ratio", 1e-6),
            "min_lr_ratio": scheduler.get("minimum_lr_ratio", 1e-6),
            "plateau_patience": scheduler.get("plateau_patience", 10),
        },
        "hardware": deepcopy(cfg["hardware"]),
        "saving": {
            "ckpt_save_also_last": bool(checkpoints.get("best", {}).get("also_save_last", False)),
            "ckpt_weights_only": bool(checkpoints.get("best", {}).get("weights_only", False)),
            "ckpt_monitor": checkpoints.get("best", {}).get("monitor", "val_miou"),
            "ckpt_monitor_mode": checkpoints.get("best", {}).get("mode", "max"),
            "early_stopping": bool(checkpoints.get("early_stopping", {}).get("enabled", False)),
            "ckpt_earlystopping_patience": checkpoints.get("early_stopping", {}).get("patience", 20),
            "cp_csv_and_conf_to_output": bool(logging.get("copy_input_csv", True)),
            "enable_progress_bar": bool(logging.get("progress", True)),
            "ckpt_verbose": bool(logging.get("checkpoint_messages", True)),
            "verbose_config": bool(logging.get("verbose_config", False)),
            "recovery_checkpoint": {
                "enabled": bool(checkpoints.get("recovery", {}).get("enabled", False)),
                "every_n_optimizer_steps": checkpoints.get("recovery", {}).get("every_steps", 2000),
                "dir_name": checkpoints.get("recovery", {}).get("directory", "recovery_checkpoints"),
            },
        },
        "_resolved_config": cfg,
        "reset_task_heads": bool(checkpoints.get("reset_task_heads", False)),
    }
    return runtime


def load_and_compile_public_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a public configuration and compile its runtime dictionary."""
    return compile_runtime_config(resolve_public_config(config))
