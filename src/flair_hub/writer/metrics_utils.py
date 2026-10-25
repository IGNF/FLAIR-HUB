import json
import numpy as np

from pathlib import Path
from typing import Dict

from flair_hub.utils.class_config import class_codes, class_weights
from flair_hub.writer.metrics_core import active_class_metrics


def compute_and_save_metrics(
    confmat: np.ndarray,
    config: Dict,
    output_dir: str,
    task: str,
    mode: str = "predict"
) -> None:
    """
    Computes segmentation evaluation metrics from a confusion matrix and saves them to disk.
    Metrics computed:
        - Per-class IoU, precision, recall, and F1-score
        - Mean IoU (mIoU)
        - Overall accuracy
        - Weighted class importance (per task and modality)
    Also logs the results in human-readable format and stores:
        - metrics.json with all results
        - confmat_<mode>.npy with raw confusion matrix
    Args:
        confmat (np.ndarray): Confusion matrix of shape (num_classes, num_classes).
        config (Dict): Configuration dictionary containing task and class metadata.
        output_dir (str): Directory where results will be saved.
        task (str): Name of the current task (used to retrieve class info).
        mode (str): Operational mode label, e.g., "predict" or "val". Used in file naming.
    Returns:
        None
    """
    label_config = config["labels_configs"][task]
    class_names = label_config["value_name"]
    num_classes = len(class_names)

    codes = class_codes(label_config)
    positions = {code: index for index, code in enumerate(codes)}
    value_weights = label_config.get('value_weights', {})
    weights = class_weights(label_config)
    default_weights = [weights[code] for code in codes]
    active_modalities = [
        mod for mod, is_active in config['modalities']["inputs"].items() if is_active
    ]
    per_modality_exceptions = value_weights.get("per_modality_exceptions", {}) or {}

    modality_weights = {}
    for mod in active_modalities:
        modality_weights[mod] = default_weights.copy()
        mod_exceptions = per_modality_exceptions.get(mod)
        if mod_exceptions:
            for i, weight in mod_exceptions.items():
                modality_weights[mod][positions[int(i)]] = weight

    weights_array = np.array(default_weights)
    used_indices = np.where(weights_array != 0)[0]

    class_names_cleaned = [class_names[codes[i]] for i in used_indices]
    default_weights_cleaned = [default_weights[i] for i in used_indices]
    modality_weights_cleaned = {
        mod: [modality_weights[mod][i] for i in used_indices]
        for mod in active_modalities
    }

    active_metrics = active_class_metrics(confmat, used_indices)
    per_c_ious = active_metrics["per_class_iou"]
    avg_ious = active_metrics["miou"]
    ovr_acc = active_metrics["overall_accuracy"]
    per_c_precision = active_metrics["per_class_precision"]
    avg_precision = active_metrics["mean_precision"]
    per_c_recall = active_metrics["per_class_recall"]
    avg_recall = active_metrics["mean_recall"]
    per_c_fscore = active_metrics["per_class_fscore"]
    avg_fscore = active_metrics["mean_fscore"]

    metrics = {
        "Avg_metrics_name": ["mIoU", "Overall Accuracy", "F-score", "Precision", "Recall"],
        "Avg_metrics": [avg_ious, ovr_acc, avg_fscore, avg_precision, avg_recall],
        "class_ids": [codes[index] for index in used_indices],
        "confusion_matrix_class_ids": codes,
        "classes": class_names_cleaned,
        "per_class_iou": list(per_c_ious),
        "per_class_fscore": list(per_c_fscore),
        "per_class_precision": list(per_c_precision),
        "per_class_recall": list(per_c_recall),
        "per_class_default_weight": default_weights_cleaned,
        "per_class_modality_weights": modality_weights_cleaned,
    }

    out_folder_metrics = Path(output_dir, f"metrics_{config['paths']['out_model_name']}", task)
    out_folder_metrics.mkdir(exist_ok=True, parents=True)
    np.save(out_folder_metrics / f"confmat_{mode}.npy", confmat)
    with open(out_folder_metrics / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTask: {task} - Global Metrics:")
    print("-" * (90 + 15 * len(active_modalities)))
    for name, value in zip(metrics["Avg_metrics_name"], metrics["Avg_metrics"]):
        print(f"{name:<20s} {value:<.4f}")
    print("-" * (90 + 15 * len(active_modalities)) + "\n")

    header = "{:<6} {:<25} {:<10} {:<10} {:<10} {:<10} {:<15}".format(
        "Idx", "Class", "IoU", "F-score", "Precision", "Recall", "w.TASK"
    )
    for mod in active_modalities:
        header += f" {'w.' + mod:<15}"
    print(header)
    print("-" * (90 + 15 * len(active_modalities)))

    for i, class_name in enumerate(class_names_cleaned):
        row = "{:<6} {:<25} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<15}".format(
            codes[used_indices[i]], class_name,
            per_c_ious[i], per_c_fscore[i],
            per_c_precision[i], per_c_recall[i],
            default_weights_cleaned[i]
        )
        for mod in active_modalities:
            row += f" {modality_weights_cleaned[mod][i]:<15}"
        print(row)
    print("\n")

    unused_indices = np.where(weights_array == 0)[0]
    if len(unused_indices) > 0:
        print("0-weighted classes for task")
        print("-" * 35)
        for idx in unused_indices:
            class_label = class_names[codes[idx]]
            print(f"{codes[idx]:<6} {class_label}")
        print("\n")
