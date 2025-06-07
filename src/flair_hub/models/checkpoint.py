import os
import torch
import torch.nn as nn

from typing import Dict, Set, List, Any
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import load_file as safe_load_file


def reinit_param(state_dict: dict, model_dict: dict, key: str) -> bool:
    """
    Reinitializes a parameter (weight or bias) in the state_dict using Xavier or zero init.
    Args:
        state_dict (dict): The state dict to modify.
        model_dict (dict): Reference model state dict.
        key (str): Parameter key to reinitialize.
    Returns:
        bool: True if key was found and reinitialized, False otherwise.
    """
    if key not in model_dict:
        return False
    with torch.no_grad():
        param = torch.empty_like(model_dict[key])
        if 'weight' in key:
            nn.init.xavier_uniform_(param)
        elif 'bias' in key:
            nn.init.zeros_(param)
        state_dict[key] = param
    return True


def interpolate_bias_table(ckpt_tensor: torch.Tensor, model_tensor: torch.Tensor) -> torch.Tensor:
    """
    Resizes a relative position bias table from a checkpoint to match the shape required by the model.
    Args:
        ckpt_tensor (torch.Tensor): Bias table from the checkpoint of shape (N_old, H).
        model_tensor (torch.Tensor): Model's expected bias table shape (N_new, H).
    Returns:
        torch.Tensor: Interpolated bias table with shape matching model_tensor.
    """
    old_len, num_heads = ckpt_tensor.shape
    new_len = model_tensor.shape[0]
    if old_len == new_len:
        return ckpt_tensor

    size_old = int(old_len ** 0.5)
    size_new = int(new_len ** 0.5)

    assert size_old * size_old == old_len, f"Checkpoint bias table shape {old_len} is not square"
    assert size_new * size_new == new_len, f"Model bias table shape {new_len} is not square"

    bias = ckpt_tensor.reshape(1, size_old, size_old, num_heads).permute(0, 3, 1, 2)
    bias = torch.nn.functional.interpolate(bias, size=(size_new, size_new), mode='bicubic', align_corners=False)
    bias = bias.permute(0, 2, 3, 1).reshape(new_len, num_heads)
    return bias


def get_task_name_from_aux_key(key: str) -> str:
    """
    Extracts the task name from a parameter key containing '__taskname'.
    """
    return key.split(".")[2].split("__")[1]


def resolve_key(key: str, state_dict: dict) -> str | None:
    """
    Resolves a parameter key by checking alternate naming conventions ('model.' prefix).
    Args:
        key (str): Original parameter key.
        state_dict (dict): State dictionary to search.
    Returns:
        str | None: Matching key in state_dict if found, else None.
    """
    candidates = [key]
    if key.startswith("model."):
        candidates.append(key[len("model."):])
    else:
        candidates.append(f"model.{key}")

    for k in candidates:
        if k in state_dict:
            return k
    return None


def check_and_reinit_layer(
    state_dict: Dict[str, torch.Tensor],
    model_dict: Dict[str, torch.Tensor],
    key_weight: str,
    key_bias: str,
    expected_classes: int,
    matched_tasks: Set[str],
    reinit_tasks: Set[str],
    task_label: str,
    reinit_counter: List[int]
) -> None:
    """
    Verifies the compatibility of a task-specific classification layer and reinitializes it if needed.

    Args:
        state_dict (Dict[str, torch.Tensor]): Checkpoint state dictionary to modify.
        model_dict (Dict[str, torch.Tensor]): Reference model's state dictionary.
        key_weight (str): Key to the classification weight tensor.
        key_bias (str): Key to the classification bias tensor.
        expected_classes (int): Expected number of output classes.
        matched_tasks (Set[str]): Set to record tasks with matching layer shapes.
        reinit_tasks (Set[str]): Set to record tasks that required reinitialization.
        task_label (str): Task name label for logging and tracking.
        reinit_counter (List[int]): Single-element list tracking number of reinitializations (mutable counter).
    """    
    real_key_weight = resolve_key(key_weight, state_dict)
    real_key_bias = resolve_key(key_bias, state_dict)

    if real_key_weight:
        ckpt_classes = state_dict[real_key_weight].shape[0]
        if ckpt_classes != expected_classes:
            print(f"→ Mismatch: {real_key_weight}: ckpt={ckpt_classes}, config={expected_classes}")
            reinit_counter[0] += reinit_param(state_dict, model_dict, key_weight)
            if real_key_bias:
                reinit_counter[0] += reinit_param(state_dict, model_dict, key_bias)
            reinit_tasks.add(task_label)
        else:
            matched_tasks.add(task_label)
    else:
        print(f"→ Missing: {key_weight}")
        if key_weight in model_dict:
            reinit_counter[0] += reinit_param(state_dict, model_dict, key_weight)
        if key_bias in model_dict:
            reinit_counter[0] += reinit_param(state_dict, model_dict, key_bias)
        reinit_tasks.add(task_label)


def strip_model_prefix_if_needed(
    state_dict: Dict[str, torch.Tensor],
    model_dict: Dict[str, torch.Tensor],
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Removes the 'model.' prefix from checkpoint keys if it's not present in the model's keys.
    Args:
        state_dict (Dict[str, torch.Tensor]): State dict loaded from a checkpoint.
        model_dict (Dict[str, torch.Tensor]): Model's state dict for comparison.
        verbose (bool): If True, prints a few stripped keys for inspection.
    Returns:
        Dict[str, torch.Tensor]: Possibly updated state_dict with adjusted keys.
    """
    sample_keys_ckpt = list(state_dict.keys())
    sample_keys_model = list(model_dict.keys())

    # Detect if 'model.' prefix mismatch
    common_key_ckpt = any(k.startswith("model.") for k in sample_keys_ckpt)
    common_key_model = all(not k.startswith("model.") for k in sample_keys_model)

    if common_key_ckpt and common_key_model:
        # Strip
        stripped_state_dict = {}
        strip_count = 0
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_k = k[len("model."):]
                stripped_state_dict[new_k] = v
                strip_count += 1
                if verbose and strip_count <= 5:
                    print(f"→ Stripping prefix: {k} → {new_k}")
            else:
                stripped_state_dict[k] = v
        if strip_count > 0:
            print(f"→ Stripped 'model.' prefix from {strip_count} keys.")
        return stripped_state_dict
    else:
        print("→ No prefix stripping needed.")
        return state_dict


@rank_zero_only
def load_checkpoint(
    conf: Dict[str, Any],
    seg_module: nn.Module,
    exit_on_fail: bool = True
) -> None:
    """
    Loads model weights from a checkpoint into the segmentation module. Handles:
    - Safetensors and PyTorch formats.
    - Shape mismatches (e.g., task heads or relative position bias).
    - Automatic reinitialization when required.
    - Partial loading with flexible prefix handling.
    Args:
        conf (Dict[str, Any]): Configuration dictionary with 'paths', 'labels', and 'labels_configs'.
        seg_module (nn.Module): Model instance into which weights will be loaded.
        exit_on_fail (bool): Whether to terminate if checkpoint path is invalid.
    Returns:
        None
    """

    print("\n" + "#" * 65)
    path = conf['paths']['ckpt_model_path']
    print(f"→ Loading checkpoint from: {path}")

    if not path or not os.path.isfile(path):
        print("❌ Invalid checkpoint path.")
        if exit_on_fail:
            raise SystemExit()
        return

    is_safe = path.endswith(".safetensors")
    if is_safe:
        print("→ Detected safetensors format.")
        state_dict = safe_load_file(path)
    else:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    print(f"→ Original state dict keys: {len(state_dict)}")

    state_dict = strip_model_prefix_if_needed(state_dict, seg_module.state_dict(), verbose=False)

    model_dict = seg_module.state_dict()
    tasks = conf["labels"]
    matched_tasks = set()
    reinit_tasks = set()
    reinit_counter = [0]

    # Main decoders
    for task in tasks:
        weight_keys = [
            f"model.main_decoders.{task}.seg_model.segmentation_head.0.weight",
            f"main_decoders.{task}.seg_model.segmentation_head.0.weight"
        ]
        bias_keys = [k.replace("weight", "bias") for k in weight_keys]
        n_classes = len(conf["labels_configs"][task]["value_name"])

        matched = False
        for w_key, b_key in zip(weight_keys, bias_keys):
            pre_match = len(matched_tasks)
            check_and_reinit_layer(state_dict, model_dict, w_key, b_key, n_classes,
                                matched_tasks, reinit_tasks, task, reinit_counter)
            if len(matched_tasks) > pre_match:
                matched = True
                break
        if not matched:
            print(f"No valid weights found for task '{task}', reinitialized.")

    # Auxiliary decoders
    for key in model_dict:
        if key.startswith("model.aux_decoders.") and "seg_model.segmentation_head.0.weight" in key:
            task_id = get_task_name_from_aux_key(key)
            n_classes = len(conf["labels_configs"].get(task_id, {}).get("value_name", []))
            bias_key = key.replace("weight", "bias")
            check_and_reinit_layer(state_dict, model_dict, key, bias_key, n_classes,
                                   matched_tasks, reinit_tasks, task_id, reinit_counter)

    # Criterion weights
    for task in tasks:
        crit_key = f"criterion.{task}.weight"
        if crit_key in state_dict and crit_key in model_dict:
            if state_dict[crit_key].shape != model_dict[crit_key].shape:
                print(f"→ Reinitializing criterion weights for {task}")
                state_dict[crit_key] = model_dict[crit_key].clone()
                reinit_counter[0] += 1

    for k in list(state_dict):
        if k in model_dict and state_dict[k].shape != model_dict[k].shape:
            ckpt_shape = state_dict[k].shape
            model_shape = model_dict[k].shape
            if "relative_position_bias_table" in k:
                print(f"→ Interpolating {k}: {ckpt_shape} → {model_shape}")
                try:
                    state_dict[k] = interpolate_bias_table(state_dict[k], model_dict[k])
                except Exception as e:
                    print(f"⚠️  Interpolation failed for {k}: {e}. Reinitializing instead.")
                    reinit_counter[0] += reinit_param(state_dict, model_dict, k)
            else:
                print(f"→ Shape mismatch for {k}: checkpoint {ckpt_shape} vs model {model_shape}. Reinitializing...")
                reinit_counter[0] += reinit_param(state_dict, model_dict, k)

    print("\nExample param BEFORE loading weights:", next(iter(seg_module.parameters())).view(-1)[:5])
    seg_module.load_state_dict(state_dict, strict=False)
    print("Example param AFTER loading weights:", next(iter(seg_module.parameters())).view(-1)[:5])

    # Summary
    print("\nCheckpoint load summary:")
    print(f"  - Tasks fully matched: {sorted(matched_tasks)}")
    print(f"  - Tasks reinitialized: {sorted(reinit_tasks)}")
    print(f"  - Total reinitialized tensors: {reinit_counter[0]}")
    print(f"  - Tasks defined in config:")
    for task in tasks:
        ncls = len(conf['labels_configs'][task]['value_name'])
        print(f"    • {task}: {ncls} classes")
    print("#" * 65 + "\n")