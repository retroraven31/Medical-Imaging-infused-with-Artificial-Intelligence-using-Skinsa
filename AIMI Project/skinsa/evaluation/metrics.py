"""
Evaluation metrics: accuracy, precision, recall, F1, AUC, confusion matrix, ROC curves.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from skinsa.config import CLASS_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, float]:
    """
    Compute all classification metrics from true labels and predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (class indices) of shape (N,).
    y_pred_probs : np.ndarray
        Predicted probabilities of shape (N, num_classes).
    class_names : list of str

    Returns
    -------
    metrics : dict with keys: accuracy, precision, recall, f1, auc,
              plus per_class_precision, per_class_recall, per_class_f1.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC (macro one-vs-rest)
    try:
        # Only include classes that appear in y_true
        present_classes = sorted(np.unique(y_true))
        roc_auc = float(roc_auc_score(
            y_true,
            y_pred_probs[:, present_classes] if y_pred_probs.shape[1] > len(present_classes) else y_pred_probs,
            multi_class="ovr",
            average="macro",
            labels=present_classes,
        ))
    except Exception:
        roc_auc = float("nan")

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(per_class_p):
            per_class[name] = {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f[i]),
            }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc,
        "per_class": per_class,
        "y_pred": y_pred.tolist(),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """Return confusion matrix as dict."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "matrix": cm.tolist(),
        "class_names": class_names,
    }


def compute_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Compute one-vs-rest ROC curves for all classes.
    Returns dict with 'curves' mapping class name -> {fpr, tpr, auc}.
    """
    n_classes = len(class_names)
    curves = {}
    for i, name in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_pred_probs[:, i])
        roc_auc = float(auc(fpr, tpr))
        curves[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc,
        }
    return {"curves": curves}


def measure_inference_time(
    model,
    images: np.ndarray,
    metadata: np.ndarray,
    warmup_runs: int = 2,
    benchmark_runs: int = 20,
) -> Dict[str, float]:
    """
    Measure per-image inference time (ms) using a small batch.

    Returns
    -------
    timing : dict with keys: mean_ms, std_ms, total_ms.
    """
    batch_imgs = images[:min(10, len(images))]
    batch_meta = metadata[:min(10, len(metadata))]

    # Warm-up runs
    for _ in range(warmup_runs):
        model.predict([batch_imgs, batch_meta], verbose=0)

    times_ms = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        model.predict([batch_imgs, batch_meta], verbose=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_image_ms = elapsed_ms / len(batch_imgs)
        times_ms.append(per_image_ms)

    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "total_ms": float(np.sum(times_ms)),
    }
