import torch
import torch.distributed as dist
import numpy as np
import rasterio
import pandas as pd

from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from flair_hub.utils.class_config import class_codes, output_dtype
from flair_hub.writer.metrics_utils import compute_and_save_metrics


def exit_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class PredictionWriter(BasePredictionWriter):
    """
    PredictionWriter callback for PyTorch Lightning.
    Handles writing segmentation predictions and evaluating metrics.
    """

    def __init__(self, config: dict, output_dir: str, write_interval: int) -> None:
        super().__init__(write_interval)
        self.config = config
        self.output_dir = output_dir
        self.accumulated_confmats = {task: None for task in config["labels"]}

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx) -> None:
        for task in self.config['labels']:
            task_config = self.config['labels_configs'][task]
            codes = class_codes(task_config)
            if self.accumulated_confmats[task] is None:
                self.accumulated_confmats[task] = np.zeros((len(codes), len(codes)), dtype=np.int64)
            preds = prediction[f'preds_{task}'].cpu().numpy().astype(output_dtype(task_config))
            for index, source in enumerate(batch[f'ID_{task}']):
                has_label = bool(batch.get(f'HAS_LABEL_{task}', [True] * len(preds))[index])
                with rasterio.open(source) as raster:
                    meta = raster.profile.copy()
                    target = raster.read(task_config.get('label_channel_nomenclature', 1)) if has_label else None
                meta.update(count=1, compress='lzw', dtype=output_dtype(task_config))
                meta.pop('nodata', None)
                meta.pop('photometric', None)
                if self.config['tasks']['write_files']:
                    folder = Path(self.output_dir, f"predictions_{self.config['paths']['out_model_name']}", task)
                    folder.mkdir(exist_ok=True, parents=True)
                    destination = folder / f'PRED_{Path(source).name}'
                    if self.config['tasks']['georeferencing_output']:
                        with rasterio.open(destination, 'w', **meta) as dst:
                            dst.write(preds[index], 1)
                    else:
                        with rasterio.open(destination, 'w', driver='GTiff', height=preds[index].shape[0],
                                           width=preds[index].shape[1], count=1, dtype=preds.dtype,
                                           compress='lzw') as dst:
                            dst.write(preds[index], 1)
                if has_label:
                    self.accumulated_confmats[task] += confusion_matrix(
                        target.flatten(), preds[index].flatten(), labels=codes)


    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        for task, local_confmat in self.accumulated_confmats.items():
            if local_confmat is None:
                task_num_classes = len(self.config["labels_configs"][task]["value_name"])
                local_confmat = np.zeros((task_num_classes, task_num_classes), dtype=int)

            tensor_confmat = torch.tensor(local_confmat, device=pl_module.device)

            if dist.is_available() and dist.is_initialized():
                gathered = [torch.zeros_like(tensor_confmat) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, tensor_confmat)

                if dist.get_rank() == 0:
                    global_confmat = sum([g.cpu().numpy() for g in gathered])
                    if global_confmat.sum():
                        compute_and_save_metrics(global_confmat, self.config, self.output_dir, task, mode="predict")
            else:
                global_confmat = tensor_confmat.cpu().numpy()
                if global_confmat.sum():
                    compute_and_save_metrics(global_confmat, self.config, self.output_dir, task, mode="predict")


    @rank_zero_only
    def load_predictions_and_compute_metrics(self) -> None:
        """
        Loads predicted label maps from disk and computes evaluation metrics for each task.
        This function:
            - Iterates over all tasks defined in the config.
            - Matches predictions to ground truth using file names.
            - Computes and accumulates confusion matrices.
            - Saves metrics (IoU, F-score, accuracy, etc.) to disk.
            - Stores valid confusion matrices in self.accumulated_confmats.
        Requires:
            - `self.config`: Dictionary with task and path metadata.
            - `self.output_dir`: Base directory containing predicted label maps.
            - `self.accumulated_confmats`: Dictionary where confusion matrices are stored.
        Side effects:
            - Writes metrics JSON and confusion matrix `.npy` files to disk.
            - Logs missing predictions or processing errors.
        Raises:
            - Prints errors for shape mismatches or file loading failures.
        Returns:
            None
        """
        any_predictions_found = False

        for task in self.config["labels"]:
            task_num_classes = len(self.config["labels_configs"][task]["value_name"])
            confmat_accum = np.zeros((task_num_classes, task_num_classes), dtype=int)

            csv_path = Path(self.config["paths"]["test_csv"])
            df = pd.read_csv(csv_path)
            gt_paths = df[task].tolist()

            pred_dir = Path(self.output_dir) / f"predictions_{self.config['paths']['out_model_name']}" / task
            missing_preds = []
            valid_preds = 0

            for gt_path in tqdm(gt_paths, desc=f"Metrics | {task}", unit="img"):
                gt_path = Path(gt_path)
                pred_path = pred_dir / f"PRED_{gt_path.name}"
                if not pred_path.exists():
                    missing_preds.append(pred_path)
                    continue

                try:
                    with rasterio.open(gt_path, 'r') as src_gt:
                        channel = self.config['labels_configs'][task].get('label_channel_nomenclature', 1)
                        gt = src_gt.read(channel)
                    with rasterio.open(pred_path, 'r') as src_pred:
                        pred = src_pred.read(1)

                    if gt.ndim == 3:
                        gt = np.squeeze(gt, axis=0)
                    if pred.ndim == 3:
                        pred = np.squeeze(pred, axis=0)

                    assert gt.shape == pred.shape, f"Shape mismatch: GT {gt.shape}, Pred {pred.shape}"


                    confmat = confusion_matrix(gt.flatten(), pred.flatten(), labels=class_codes(self.config["labels_configs"][task]))
                    confmat_accum += confmat
                    valid_preds += 1

                except Exception as e:
                    print(f"[ERROR] Failed to process {gt_path.name}: {e}")

            print(f"Confmat sum: {confmat_accum.sum()}")
            print(f"Total GT images processed: {valid_preds} / {len(gt_paths)}")

            if valid_preds > 0:
                self.accumulated_confmats[task] = confmat_accum
                compute_and_save_metrics(confmat_accum, self.config, self.output_dir, task, mode="metrics_only")
                any_predictions_found = True

        if not any_predictions_found:
            print("\n[ERROR] No predictions found at all. Metrics will not be calculated.\n")

        exit_ddp()
