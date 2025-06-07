import numpy as np


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
