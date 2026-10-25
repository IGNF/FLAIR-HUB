import os
import torch
import torch.nn as nn

from typing import Dict, Set, List, Any
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import load_file as safe_load_file
from flair_hub.utils.class_config import model_num_classes, model_class_indices, nomenclature_signature


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


    common_key_ckpt = any(k.startswith("model.") for k in sample_keys_ckpt)
    common_key_model = all(not k.startswith("model.") for k in sample_keys_model)

    if common_key_ckpt and common_key_model:

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


def load_checkpoint(conf, seg_module, exit_on_fail=True):
    """Strict restoration for inference, explicit partial loading for fine-tuning.

    Every distributed rank loads its weights. Only reporting may be rank-zero.
    """
    path = conf['paths']['ckpt_model_path']
    if not path or not os.path.isfile(path):
        if exit_on_fail:
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return
    initializing = bool(conf.get('tasks', {}).get('train', False) and
                        conf.get('tasks', {}).get('train_tasks', {}).get(
                            'init_weights_only_from_ckpt', False))
    if path.endswith('.safetensors'):
        checkpoint = {}
        state = safe_load_file(path)
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
    model_dict = seg_module.state_dict()
    state = strip_model_prefix_if_needed(state, model_dict)
    normalized = {}
    for key, value in state.items():
        target = next((candidate for candidate in (
            key, f"model.{key}", key.removeprefix("model.")
        ) if candidate in model_dict), key)
        if target in normalized:
            raise ValueError(f"Ambiguous checkpoint keys resolve to {target}")
        normalized[target] = value
    state = normalized
    saved_codes = checkpoint.get('output_class_codes')
    saved_names = checkpoint.get('nomenclatures')
    current_codes = {t: model_class_indices(conf['labels_configs'][t]) for t in conf['labels']}
    changed = {t for t, codes in current_codes.items()
               if (saved_codes.get(t) if saved_codes is not None else list(range(len(codes)))) != codes}
    if saved_names is not None:
        changed.update(t for t in current_codes
                       if saved_names.get(t) != nomenclature_signature(conf['labels_configs'][t]))
    if conf.get('reset_task_heads', False) and initializing:
        changed.update(current_codes)
    if changed and not initializing:
        raise ValueError(f'Checkpoint nomenclature differs for {sorted(changed)}; use matching artifacts')

    skipped = []
    for key in list(state):
        bare = key.removeprefix('model.')
        task_output = any(bare.startswith(f'main_decoders.{t}.') or
                          (bare.startswith('aux_decoders.') and f'__{t}.' in bare)
                          for t in changed)
        output_projection = ('.classifier.' in bare or '.segmentation_head.' in bare or
                             (task_output and bare.count('.') == 2))
        reset = (task_output and output_projection) or (
            bool(changed) and bare.startswith('encoders.') and '.out_conv.' in bare)
        if bare.startswith('criterion.') or reset:
            skipped.append(key)
            del state[key]
        elif key in model_dict and state[key].shape != model_dict[key].shape:
            if not initializing:
                raise ValueError(f'Checkpoint tensor shape mismatch: {key}')
            if 'relative_position_bias_table' in key:
                try:
                    state[key] = interpolate_bias_table(state[key], model_dict[key])
                    continue
                except (ValueError, AssertionError, RuntimeError):
                    pass
            skipped.append(key)
            del state[key]

    missing = [k for k in model_dict if k not in state and not k.startswith('criterion.')]
    unexpected = [k for k in state if k not in model_dict and
                  not k.removeprefix('model.').startswith('aux_decoders.')]
    if not initializing and (missing or unexpected):
        raise ValueError(f'Incomplete checkpoint: missing={missing[:12]}, unexpected={unexpected[:12]}')
    matched = [key for key in state if key in model_dict]
    if not matched:
        raise ValueError('Checkpoint contains no compatible model tensors')
    seg_module.load_state_dict(state, strict=False)
    print(f'[CHECKPOINT] Loaded {len(matched)} tensors; '
          f'fresh={len(missing)}, skipped={len(skipped)}, unexpected={len(unexpected)}; '
          f'fine_tuning={initializing}')
