"""
Classification evaluation utilities: confusion matrix and a per-class
precision/recall/F1 report, generalizing train.py's binary-only
compute_precision_recall_f1 to the multi-class case.
"""
from typing import Any, Dict, Optional

from ..core.backend import np
from ..core import backend


def confusion_matrix(y_true: Any, y_pred: Any, num_classes: Optional[int] = None) -> Any:
    """Return an (num_classes, num_classes) matrix where entry [i, j] counts
    samples with true class i predicted as class j. y_true/y_pred: 1D
    integer class-label arrays (use argmax first for one-hot/softmax output)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("confusion_matrix: y_true/y_pred must be non-empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"confusion_matrix: y_true has shape {y_true.shape} but y_pred has "
            f"shape {y_pred.shape} -- they must match."
        )
    observed_max = int(max(y_true.max(), y_pred.max()))
    if num_classes is None:
        num_classes = observed_max + 1
    elif observed_max >= num_classes:
        raise ValueError(
            f"confusion_matrix: num_classes={num_classes} but y_true/y_pred contain "
            f"a label as large as {observed_max} -- pass a larger num_classes (or "
            f"omit it to infer from the data)."
        )
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report(y_true: Any, y_pred: Any, num_classes: Optional[int] = None) -> Dict[Any, Any]:
    """Per-class precision/recall/f1/support plus macro and weighted averages.

    Returns a dict: {class_idx: {precision, recall, f1, support}, ...,
    "macro_avg": {...}, "weighted_avg": {...}, "accuracy": float}.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    n_classes = cm.shape[0]
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    predicted_positives = cm.sum(axis=0)

    # np.divide's `where=` kwarg (used to avoid 0/0 without a separate mask
    # step) isn't supported by CuPy's ufuncs, so these do an explicit
    # boolean-mask assignment instead -- works identically on both backends.
    precision = np.zeros(n_classes, dtype=backend.default_dtype())
    mask = predicted_positives != 0
    precision[mask] = tp[mask] / predicted_positives[mask]

    recall = np.zeros(n_classes, dtype=backend.default_dtype())
    mask = support != 0
    recall[mask] = tp[mask] / support[mask]

    f1_denom = precision + recall
    f1 = np.zeros(n_classes, dtype=backend.default_dtype())
    mask = f1_denom != 0
    f1[mask] = (2 * precision * recall)[mask] / f1_denom[mask]

    report = {}
    for c in range(n_classes):
        report[c] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }

    total_support = support.sum()
    report["accuracy"] = float(tp.sum() / total_support) if total_support > 0 else 0.0
    report["macro_avg"] = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "support": int(total_support),
    }
    weights = support / total_support if total_support > 0 else np.zeros(n_classes)
    report["weighted_avg"] = {
        "precision": float(np.sum(precision * weights)),
        "recall": float(np.sum(recall * weights)),
        "f1": float(np.sum(f1 * weights)),
        "support": int(total_support),
    }
    return report
