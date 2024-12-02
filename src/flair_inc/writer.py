from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio
from PIL import Image
import numpy as np
import json
import shutil
import pandas as pd
from sklearn.metrics import confusion_matrix
from pytorch_lightning.utilities.rank_zero import rank_zero_only



def overall_accuracy(npcm):
    """
    Calculate the overall accuracy from the normalized confusion matrix (NPCM).
    Args:
        npcm (ndarray): Normalized confusion matrix.
    Returns:
        float: Overall accuracy as a percentage.
    """
    oa = np.trace(npcm) / npcm.sum()
    return 100 * oa


def class_IoU(npcm, n_class):
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


def class_precision(npcm):
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


def class_recall(npcm):
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


def class_fscore(precision, recall):
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




class PredictionWriter(BasePredictionWriter):
    """
    A class to write predictions to disk and calculate metrics from confusion matrices.
    Args:
        config (dict): Configuration settings.
        output_dir (str): Directory where output files will be saved.
        write_interval (int): Interval at which predictions are written.
    """
    def __init__(self, config, output_dir, write_interval):
        super().__init__(write_interval)

        self.config = config
        self.output_dir = output_dir
        self.accumulated_confmat = None

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch,
        batch_idx, dataloader_idx
    ):
        """
        Write predictions to disk at the end of a batch and accumulate confusion matrices.
        Args:
            trainer (pl.Trainer): The current trainer.
            pl_module (pl.LightningModule): The current module.
            prediction (dict): A dictionary containing predictions and IDs.
            batch_indices (list): List of indices for the current batch.
            batch (dict): The current batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the current dataloader.
        """
        preds, filenames = prediction["preds"], prediction["ID"]
        preds = preds.cpu().numpy().astype('uint8')  # Move prediction to CPU

        if self.config['tasks']['write_files']:
            output_dir_predictions = Path(self.output_dir, 'predictions')
            output_dir_predictions.mkdir(exist_ok=True, parents=True)

            if self.config['georeferencing_output']:
                for prediction, filename in zip(preds, filenames):
                    output_file = str(output_dir_predictions / 
                                      f'PRED_{filename.split("/")[-1]}')
                    with rasterio.open(filename, 'r') as f:
                        meta = f.profile  # Extract georeferencing info from input image
                        meta['count'] = 1
                        meta['compress'] = 'lzw'
                    with rasterio.open(output_file, 'w', **meta) as dst:
                        dst.write(prediction, 1)
            else:
                for prediction, filename in zip(preds, filenames):
                    output_file = str(output_dir_predictions / 
                                      f'PRED_{filename.split("/")[-1]}')
                    Image.fromarray(prediction).save(output_file, compression='tiff_lzw')

        # Accumulate confusion matrices
        for pred, gt_mask in zip(preds, filenames):
            target = np.array(Image.open(gt_mask)) - 1
            confmat = confusion_matrix(
                target.flatten(), pred.flatten(), labels=list(range(len(self.config["classes"])))
            )
            if self.accumulated_confmat is None:
                self.accumulated_confmat = confmat
            else:
                self.accumulated_confmat += confmat


    def on_predict_epoch_end(self, trainer, pl_module):
        """
        Callback function to calculate metrics at the end of the prediction epoch.
        Args:
            trainer (pl.Trainer): The current trainer.
            pl_module (pl.LightningModule): The current module.
        """
        self.calculate_metrics()


    @rank_zero_only
    def calculate_metrics(self):
        """
        Calculate and log metrics based on the accumulated confusion matrix.
        These include IoU, F-score, Precision, and Recall per class and overall.
        """
        if self.accumulated_confmat is None:
            print("No confusion matrix accumulated.")
            return

        # Calculate metrics
        weights = np.array([self.config["classes"][i][0] for i in self.config["classes"]])
        unused_classes = np.where(weights == 0)[0]
        confmat_cleaned = np.delete(self.accumulated_confmat, unused_classes, axis=0)  # Remove rows
        confmat_cleaned = np.delete(confmat_cleaned, unused_classes, axis=1)  # Remove columns

        per_c_ious, avg_ious = class_IoU(confmat_cleaned, len(np.nonzero(weights)[0]))
        ovr_acc = overall_accuracy(confmat_cleaned)
        per_c_precision, avg_precision = class_precision(confmat_cleaned)
        per_c_recall, avg_recall = class_recall(confmat_cleaned)
        per_c_fscore, avg_fscore = class_fscore(per_c_precision, per_c_recall)

        metrics = {
            'Avg_metrics_name': ['mIoU', 'Overall Accuracy', 'Fscore', 'Precision', 'Recall'],
            'Avg_metrics': [avg_ious, ovr_acc, avg_fscore, avg_precision, avg_recall],
            'classes': list(np.array([self.config["classes"][i][1] for i in self.config["classes"]])[np.nonzero(weights)[0]]),
            'per_class_iou': list(per_c_ious),
            'per_class_fscore': list(per_c_fscore),
            'per_class_precision': list(per_c_precision),
            'per_class_recall': list(per_c_recall),
        }

        out_folder_metrics = Path(self.output_dir, 'metrics')
        out_folder_metrics.mkdir(exist_ok=True, parents=True)
        np.save(out_folder_metrics / 'confmat.npy', self.accumulated_confmat)
        json.dump(metrics, open(out_folder_metrics / 'metrics.json', 'w'))

        print('')
        print('Global Metrics: ')
        print('-' * 90)
        for metric_name, metric_value in zip(metrics['Avg_metrics_name'], metrics['Avg_metrics']):
            print(f"{metric_name:<20s} {metric_value:<20.4f}")
        print('-' * 90 + '\n\n')

        # Separate classes into used and unused based on weight
        used_classes = {k: v for k, v in self.config["classes"].items() if v[0] != 0}
        unused_classes = {k: v for k, v in self.config["classes"].items() if v[0] == 0}

        def print_class_metrics(class_dict, metrics_available=True):
            """
            Print class-specific metrics.
            Args:
                class_dict (dict): Dictionary of class metrics.
                metrics_available (bool): Flag to indicate if metrics are available.
            """
            for class_index, class_info in class_dict.items():
                class_weight, class_name = class_info
                if metrics_available:
                    i = metrics['classes'].index(class_name)  # Get the index of the class in the metrics
                    print("{:<25} {:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        class_name, class_weight, metrics['per_class_iou'][i],
                        metrics['per_class_fscore'][i], metrics['per_class_precision'][i],
                        metrics['per_class_recall'][i]))
                else:
                    print("{:<25} {:<15}".format(class_name, class_weight))

        print("{:<25} {:<15} {:<10} {:<10} {:<10} {:<10}".format('Class', 'Weight', 'IoU', 'F-score', 'Precision', 'Recall'))
        print('-' * 65)
        print_class_metrics(used_classes)
        print("\nNot learned Classes:")
        print("{:<25} {:<15}".format('Class', 'Weight'))
        print('-' * 65)
        print_class_metrics(unused_classes, metrics_available=False)
        print('\n\n')


    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Callback function to handle batch end during prediction.
        Args:
            trainer (pl.Trainer): The current trainer.
            pl_module (pl.LightningModule): The current module.
            outputs (dict): Outputs from the current batch.
            batch (dict): The current batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader.
        """
        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )
