import rasterio
import numpy as np
import json

from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
from PIL import Image
from sklearn.metrics import confusion_matrix
from pytorch_lightning.utilities.rank_zero import rank_zero_only



class PredictionWriter(BasePredictionWriter):
    """
    Multi-task PredictionWriter for writing predictions and computing metrics.

    Args:
        config (dict): Model configuration.
        output_dir (str): Directory to save outputs.
        write_interval (int): Interval for writing predictions.
    """

    def __init__(self, config: dict, output_dir: str, write_interval: int) -> None:
        super().__init__(write_interval)

        self.config = config
        self.output_dir = output_dir
        self.accumulated_confmats = {task: None for task in config["labels"]}

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx) -> None:
        """
        Write predictions and accumulate confusion matrices per task.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.
            prediction: The prediction made by the model.
            batch_indices: Indices of the current batch.
            batch: The current batch of data.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the dataloader.
        """
        for task in self.config['labels']:
            id_in_file = batch[f'ID_{task}']
            task_num_classes = len(self.config["labels_configs"][task]["value_name"])

            # Prepare metrics / output
            if self.accumulated_confmats[task] is None:
                self.accumulated_confmats[task] = np.zeros((task_num_classes, task_num_classes), dtype=int)
            
            output_dir_predictions = Path(self.output_dir, "predictions", task)
            output_dir_predictions.mkdir(exist_ok=True, parents=True)

            # Gather predictions and ground truth
            preds = prediction[f'preds_{task}'].cpu().numpy().astype("uint8")
            self.channels = None
            if 'label_channel_nomenclature' in self.config['labels_configs'][task]:
                self.channels = [self.config['labels_configs'][task]['label_channel_nomenclature']]

            with rasterio.open(id_in_file[0], 'r') as src_img:
                target = src_img.read(self.channels)

            # Write predictions if triggered
            if self.config["tasks"]["write_files"]:
                if self.config["tasks"]["georeferencing_output"]:
                    out_name = f"PRED_{id_in_file[0].split('/')[-1]}"
                    output_file = str(output_dir_predictions / out_name)
                    with rasterio.open(id_in_file[0], "r") as f:
                        meta = f.profile
                        meta["count"] = 1
                        meta["compress"] = "lzw"
                    with rasterio.open(output_file, "w", **meta) as dst:
                        dst.write(preds[0].astype("uint8"), 1)
                else:
                    out_name = f"PRED_{id_in_file[0].split('/')[-1]}"
                    output_file = str(output_dir_predictions / out_name)
                    Image.fromarray(preds[0]).save(output_file, compression="tiff_lzw")

            # Calculate metrics
            confmat = confusion_matrix(target.flatten(), preds[0].flatten(), labels=list(range(task_num_classes)))
            self.accumulated_confmats[task] += confmat

    @rank_zero_only
    def calculate_metrics(self) -> None:
        """
        Compute and save metrics for each task.

        Args:
            None

        Returns:
            None
        """
        for task, confmat in self.accumulated_confmats.items():
            if confmat is None:
                print(f"No confusion matrix accumulated for task {task}.")
                continue

            label_config = self.config["labels_configs"][task]
            class_names = label_config["value_name"]
            num_classes = len(class_names)

            per_c_ious, avg_ious = class_IoU(confmat, num_classes)
            ovr_acc = overall_accuracy(confmat)
            per_c_precision, avg_precision = class_precision(confmat)
            per_c_recall, avg_recall = class_recall(confmat)
            per_c_fscore, avg_fscore = class_fscore(per_c_precision, per_c_recall)

            metrics = {
                "Avg_metrics_name": ["mIoU", "Overall Accuracy", "F-score", "Precision", "Recall"],
                "Avg_metrics": [avg_ious, ovr_acc, avg_fscore, avg_precision, avg_recall],
                "classes": list(class_names.values()),
                "per_class_iou": list(per_c_ious),
                "per_class_fscore": list(per_c_fscore),
                "per_class_precision": list(per_c_precision),
                "per_class_recall": list(per_c_recall),
            }

            out_folder_metrics = Path(self.output_dir, "metrics", task)
            out_folder_metrics.mkdir(exist_ok=True, parents=True)
            np.save(out_folder_metrics / "confmat.npy", confmat)
            json.dump(metrics, open(out_folder_metrics / "metrics.json", "w"))

            print(f"\nTask: {task} - Global Metrics:")
            print("-" * 90)
            for metric_name, metric_value in zip(metrics["Avg_metrics_name"], metrics["Avg_metrics"]):
                print(f"{metric_name:<20s} {metric_value:<20.4f}")
            print("-" * 90 + "\n")

            # Print per-class metrics
            print("{:<6} {:<25} {:<10} {:<10} {:<10} {:<10}".format("Idx", "Class", "IoU", "F-score", "Precision", "Recall"))
            print("-" * 75)
            for i, class_name in enumerate(class_names.values()):
                print("{:<6} {:<25} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    i,
                    class_name,
                    per_c_ious[i],
                    per_c_fscore[i],
                    per_c_precision[i],
                    per_c_recall[i]
                ))
            print("\n")

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        """
        Compute metrics at the end of the prediction epoch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.

        Returns:
            None
        """
        self.calculate_metrics()






def overall_accuracy(npcm: np.ndarray) -> float:
    """
    Calculate the overall accuracy from the normalized confusion matrix (NPCM).

    Args:
        npcm (ndarray): Normalized confusion matrix.

    Returns:
        float: Overall accuracy as a percentage.
    """
    oa = np.trace(npcm) / npcm.sum()
    return 100 * oa


def class_IoU(npcm: np.ndarray, n_class: int) -> tuple:
    """
    Calculate the Intersection over Union (IoU) for each class and the mean IoU.

    Args:
        npcm (ndarray): Normalized confusion matrix.
        n_class (int): Number of classes.

    Returns:
        tuple: (IoUs for each class as an array, mean IoU as a float).
    """
    ious = 100 * np.diag(npcm) / (np.sum(npcm, axis=1) + np.sum(npcm, axis=0) - np.diag(npcm))
    ious[np.isnan(ious)] = 0
    return ious, np.mean(ious)


def class_precision(npcm: np.ndarray) -> tuple:
    """
    Calculate the precision for each class and the mean precision.

    Args:
        npcm (ndarray): Normalized confusion matrix.

    Returns:
        tuple: (Precision for each class as an array, mean precision as a float).
    """
    precision = 100 * np.diag(npcm) / np.sum(npcm, axis=0)
    precision[np.isnan(precision)] = 0
    return precision, np.mean(precision)


def class_recall(npcm: np.ndarray) -> tuple:
    """
    Calculate the recall for each class and the mean recall.

    Args:
        npcm (ndarray): Normalized confusion matrix.

    Returns:
        tuple: (Recall for each class as an array, mean recall as a float).
    """
    recall = 100 * np.diag(npcm) / np.sum(npcm, axis=1)
    recall[np.isnan(recall)] = 0
    return recall, np.mean(recall)


def class_fscore(precision: np.ndarray, recall: np.ndarray) -> tuple:
    """
    Calculate the F-score for each class and the mean F-score.

    Args:
        precision (ndarray): Precision for each class.
        recall (ndarray): Recall for each class.

    Returns:
        tuple: (F-score for each class as an array, mean F-score as a float).
    """
    fscore = 2 * (precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    return fscore, np.mean(fscore)