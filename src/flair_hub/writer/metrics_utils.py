import json
import numpy as np

from pathlib import Path
from typing import Dict

from flair_hub.writer.metrics_core import (
                                            class_IoU,
                                            overall_accuracy,
                                            class_precision,
                                            class_recall,
                                            class_fscore
)


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

    value_weights = label_config.get("value_weights", {})
    default_weight = value_weights.get("default", 1)
    default_exceptions = value_weights.get("default_exceptions", {}) or {}
    default_weights = [default_weight] * num_classes
    for i, weight in default_exceptions.items():
        default_weights[i] = weight

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
                modality_weights[mod][i] = weight

    weights_array = np.array(default_weights)
    used_indices = np.where(weights_array != 0)[0]

    confmat_cleaned = confmat[np.ix_(used_indices, used_indices)]
    class_names_cleaned = [class_names[i] for i in used_indices]
    default_weights_cleaned = [default_weights[i] for i in used_indices]
    modality_weights_cleaned = {
        mod: [modality_weights[mod][i] for i in used_indices]
        for mod in active_modalities
    }

    per_c_ious, avg_ious = class_IoU(confmat_cleaned, len(used_indices))
    ovr_acc = overall_accuracy(confmat_cleaned)
    per_c_precision, avg_precision = class_precision(confmat_cleaned)
    per_c_recall, avg_recall = class_recall(confmat_cleaned)
    per_c_fscore, avg_fscore = class_fscore(per_c_precision, per_c_recall)

    metrics = {
        "Avg_metrics_name": ["mIoU", "Overall Accuracy", "F-score", "Precision", "Recall"],
        "Avg_metrics": [avg_ious, ovr_acc, avg_fscore, avg_precision, avg_recall],
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
            i, class_name,
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
            class_label = class_names[idx]
            print(f"{idx:<6} {class_label}")
        print("\n")
