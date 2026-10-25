import numpy as np


def active_class_metrics(
    confmat: np.ndarray,
    active_indices: np.ndarray,
) -> dict:
    """Compute metrics on active GT classes without hiding invalid predictions.

    Rows are ground-truth classes and columns are predicted classes. Ground-
    truth rows excluded by the task weights are ignored entirely. All prediction
    columns remain in the row totals, so predicting an excluded class for an
    active target is still counted as a false negative.
    """
    confmat = np.asarray(confmat, dtype=np.float64)
    active_indices = np.asarray(active_indices, dtype=np.int64)
    if confmat.ndim != 2 or confmat.shape[0] != confmat.shape[1]:
        raise ValueError("confmat must be a square matrix")
    if active_indices.size == 0:
        raise ValueError("at least one active class is required")
    if active_indices.min() < 0 or active_indices.max() >= confmat.shape[0]:
        raise ValueError("active class index is outside the confusion matrix")

    active_target_rows = confmat[active_indices, :]
    true_positives = confmat[active_indices, active_indices]
    target_totals = active_target_rows.sum(axis=1)
    prediction_totals = active_target_rows[:, active_indices].sum(axis=0)

    def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(numerator, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator != 0)
        return 100.0 * result

    iou = safe_ratio(
        true_positives,
        target_totals + prediction_totals - true_positives,
    )
    precision = safe_ratio(true_positives, prediction_totals)
    recall = safe_ratio(true_positives, target_totals)
    fscore = np.zeros_like(precision)
    np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=fscore,
        where=(precision + recall) != 0,
    )
    active_pixel_count = target_totals.sum()
    overall = (
        100.0 * true_positives.sum() / active_pixel_count
        if active_pixel_count
        else 0.0
    )
    return {
        "per_class_iou": iou,
        "miou": float(iou.mean()),
        "overall_accuracy": float(overall),
        "per_class_precision": precision,
        "mean_precision": float(precision.mean()),
        "per_class_recall": recall,
        "mean_recall": float(recall.mean()),
        "per_class_fscore": fscore,
        "mean_fscore": float(fscore.mean()),
    }


def overall_accuracy(npcm: np.ndarray) -> float:
    """
    Calculate the overall accuracy from the normalized confusion matrix (NPCM).
    """
    oa = np.trace(npcm) / npcm.sum()
    return 100 * oa


def class_IoU(npcm: np.ndarray, n_class: int) -> tuple:
    """
    Calculate the Intersection over Union (IoU) for each class and the mean IoU.
    """
    ious = 100 * np.diag(npcm) / (np.sum(npcm, axis=1) + np.sum(npcm, axis=0) - np.diag(npcm))
    ious[np.isnan(ious)] = 0
    return ious, np.mean(ious)


def class_precision(npcm: np.ndarray) -> tuple:
    """
    Calculate the precision for each class and the mean precision.
    """
    precision = 100 * np.diag(npcm) / np.sum(npcm, axis=0)
    precision[np.isnan(precision)] = 0
    return precision, np.mean(precision)


def class_recall(npcm: np.ndarray) -> tuple:
    """
    Calculate the recall for each class and the mean recall.
    """
    recall = 100 * np.diag(npcm) / np.sum(npcm, axis=1)
    recall[np.isnan(recall)] = 0
    return recall, np.mean(recall)


def class_fscore(precision: np.ndarray, recall: np.ndarray) -> tuple:
    """
    Calculate the F-score for each class and the mean F-score.
    """
    fscore = 2 * (precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    return fscore, np.mean(fscore)
