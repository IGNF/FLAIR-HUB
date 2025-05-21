import os
import torch

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import load_file as safe_load_file

def reinit_param(state_dict, model_dict, key):
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

def get_task_name_from_aux_key(key: str) -> str:
    return key.split(".")[2].split("__")[1]

def resolve_key(key, state_dict):
    # Try raw, stripped, or prefixed key
    candidates = [key, key.removeprefix("model."), f"model.{key}"]
    for k in candidates:
        if k in state_dict:
            return k
    return None

def check_and_reinit_layer(state_dict, model_dict, key_weight, key_bias, expected_classes, matched_tasks, reinit_tasks, task_label, reinit_counter):
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

def strip_model_prefix_if_needed(state_dict, model_dict, verbose=False):
    sample_keys_ckpt = list(state_dict.keys())
    sample_keys_model = list(model_dict.keys())

    # Detect if 'model.' prefix mismatch
    common_key_ckpt = any(k.startswith("model.") for k in sample_keys_ckpt)
    common_key_model = all(not k.startswith("model.") for k in sample_keys_model)

    if common_key_ckpt and common_key_model:
        # Strip safely
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
def load_checkpoint(conf, seg_module, exit_on_fail=False):
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
        w_key = f"main_decoders.{task}.seg_model.segmentation_head.0.weight"
        b_key = w_key.replace("weight", "bias")
        n_classes = len(conf["labels_configs"][task]["value_name"])
        check_and_reinit_layer(state_dict, model_dict, w_key, b_key, n_classes,
                               matched_tasks, reinit_tasks, task, reinit_counter)

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

    # Shape mismatch fallback
    for k in list(state_dict):
        if k in model_dict and state_dict[k].shape != model_dict[k].shape:
            print(f"→ Shape mismatch for {k}. Reinitializing...")
            reinit_counter[0] += reinit_param(state_dict, model_dict, k)

    # Debug values
    print("\nExample param BEFORE:", next(iter(seg_module.parameters())).view(-1)[:5])

    # Load
    seg_module.load_state_dict(state_dict, strict=False)

    print("Example param AFTER:", next(iter(seg_module.parameters())).view(-1)[:5])

    # Summary
    print("\n✅ Checkpoint load summary:")
    print(f"  - Tasks fully matched: {sorted(matched_tasks)}")
    print(f"  - Tasks reinitialized: {sorted(reinit_tasks)}")
    print(f"  - Total reinitialized tensors: {reinit_counter[0]}")
    print(f"  - Tasks defined in config:")
    for task in tasks:
        ncls = len(conf['labels_configs'][task]['value_name'])
        print(f"    • {task}: {ncls} classes")
    print("#" * 65 + "\n")
