import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true, logits):
    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {logits.shape}")
    if len(y_true) != logits.shape[0]:
        raise ValueError(f"y_true and logits contain different numbers of samples: {len(y_true)} != {logits.shape[0]}")

    y_pred = np.argmax(logits, axis=1)
    labels = list(range(logits.shape[1]))
    n_classes = len(labels)
    if np.any((y_true < 0) | (y_true >= n_classes)):
        raise ValueError(f"y_true contains labels outside 0..{n_classes - 1}")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values")

    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    weighted_f1 = metrics.f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    class_f1 = metrics.f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    confusion_mat = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    class_ovr_accuracy = []
    for cls in range(n_classes):
        true_binary = (y_true == cls).astype(int)
        pred_binary = (y_pred == cls).astype(int)
        class_ovr_accuracy.append(np.mean(true_binary == pred_binary))

    y_true_bin = label_binarize(y_true, classes=labels)
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_pred_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    class_auroc = []
    for cls in range(n_classes):
        positives = int(y_true_bin[:, cls].sum())
        negatives = int(len(y_true_bin) - positives)
        class_auroc.append(
            float(metrics.roc_auc_score(y_true_bin[:, cls], y_pred_proba[:, cls]))
            if positives > 0 and negatives > 0
            else float("nan")
        )
    valid_auc = np.isfinite(class_auroc)
    if not np.any(valid_auc):
        raise ValueError("AUROC is undefined because no class has both positive and negative samples")
    macro_auroc = float(np.nanmean(class_auroc))
    class_counts = np.bincount(y_true, minlength=n_classes).astype(float)
    weighted_auroc = float(np.average(np.asarray(class_auroc)[valid_auc], weights=class_counts[valid_auc]))

    return {
        "overall_accuracy": overall_accuracy,
        "class_ovr_accuracy": class_ovr_accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "class_f1": class_f1,
        "confusion_mat": confusion_mat,
        "class_auroc": class_auroc,
        "macro_auroc": macro_auroc,
        "weighted_auroc": weighted_auroc,
        "y_true_bin": y_true_bin,
        "y_pred_proba": y_pred_proba,
    }
