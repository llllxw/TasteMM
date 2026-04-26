import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true, logits):
    y_pred = np.argmax(logits, axis=1)
    labels = [0, 1, 2, 3, 4, 5]
    total_samples = len(y_true)
    n_classes = len(labels)

    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    weighted_f1 = metrics.f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    class_f1 = metrics.f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    confusion_mat = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    class_binary_accuracy = []
    for cls in range(n_classes):
        true_binary = (y_true == cls).astype(int)
        pred_binary = (y_pred == cls).astype(int)
        class_binary_accuracy.append(np.mean(true_binary == pred_binary))

    try:
        y_true_bin = label_binarize(y_true, classes=labels)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_pred_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        class_auroc = []
        for cls in range(n_classes):
            if int(y_true_bin[:, cls].sum()) == 0:
                class_auroc.append(0.0)
            else:
                class_auroc.append(metrics.roc_auc_score(y_true_bin[:, cls], y_pred_proba[:, cls]))
        macro_auroc = float(np.mean(class_auroc))
        class_counts = np.bincount(y_true, minlength=n_classes)
        weighted_auroc = float(np.average(class_auroc, weights=class_counts))
    except Exception:
        class_auroc = [0.0] * n_classes
        macro_auroc = 0.0
        weighted_auroc = 0.0
        y_true_bin = np.zeros((total_samples, n_classes))
        y_pred_proba = np.zeros((total_samples, n_classes))

    return {
        "overall_accuracy": overall_accuracy,
        "class_binary_accuracy": class_binary_accuracy,
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
