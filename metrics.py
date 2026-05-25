import numpy as np


def compute_precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_f1(y_true, y_pred):
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_roc_auc(y_true, y_probs):
    desc_score_indices = np.argsort(y_probs)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc
