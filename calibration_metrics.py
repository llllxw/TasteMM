import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def ece_top1(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = (conf > bins[idx]) & (conf <= bins[idx + 1])
        if np.any(mask):
            ece += abs(float(acc[mask].mean()) - float(conf[mask].mean())) * float(mask.mean())
    return float(ece)


def reliability_bins_top1(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> dict:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    out = {
        "bin_lower": bins[:-1],
        "bin_upper": bins[1:],
        "count": [],
        "fraction": [],
        "accuracy": [],
        "confidence": [],
    }
    total = max(len(y_true), 1)
    for idx in range(n_bins):
        mask = (conf > bins[idx]) & (conf <= bins[idx + 1])
        count = int(mask.sum())
        out["count"].append(count)
        out["fraction"].append(float(count / total))
        out["accuracy"].append(float(correct[mask].mean()) if count else float("nan"))
        out["confidence"].append(float(conf[mask].mean()) if count else float("nan"))
    return out


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_onehot = np.zeros_like(probs, dtype=np.float32)
    y_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def brier_top1(probs: np.ndarray, y_true: np.ndarray) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.int32)
    return float(np.mean((conf - correct) ** 2))


def classwise_brier_score(probs: np.ndarray, y_true: np.ndarray) -> dict:
    out = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        out[int(cls)] = brier_score(probs[mask], y_true[mask]) if mask.sum() >= 5 else float("nan")
    return out


def classwise_ece_top1(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> dict:
    out = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        out[int(cls)] = ece_top1(probs[mask], y_true[mask], n_bins=n_bins) if mask.sum() >= 5 else float("nan")
    return out


def classwise_brier_top1(probs: np.ndarray, y_true: np.ndarray) -> dict:
    out = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        out[int(cls)] = brier_top1(probs[mask], y_true[mask]) if mask.sum() >= 5 else float("nan")
    return out


def risk_coverage_curve(conf: np.ndarray, correct01: np.ndarray):
    order = np.argsort(-conf)
    sorted_correct = correct01[order].astype(np.float32)
    return risk_coverage_curve_from_sorted_correct(sorted_correct)


def risk_coverage_curve_from_sorted_correct(sorted_correct: np.ndarray):
    n = len(sorted_correct)
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    kept = np.arange(1, n + 1)
    acc = np.cumsum(sorted_correct) / kept
    risk = 1.0 - acc
    coverage = kept / n
    return np.concatenate([[0.0], coverage]), np.concatenate([[0.0], risk])


def aurc(conf: np.ndarray, correct01: np.ndarray) -> float:
    coverage, risk = risk_coverage_curve(conf, correct01)
    return _trapezoid(risk, coverage)


def optimal_aurc(correct01: np.ndarray) -> float:
    sorted_correct = np.sort(correct01.astype(np.float32))[::-1]
    coverage, risk = risk_coverage_curve_from_sorted_correct(sorted_correct)
    return _trapezoid(risk, coverage)


def eaurc(conf: np.ndarray, correct01: np.ndarray) -> float:
    return float(max(aurc(conf, correct01) - optimal_aurc(correct01), 0.0))


def risk_at_coverage(conf: np.ndarray, correct01: np.ndarray, target_cov: float = 0.8) -> float:
    coverage, risk = risk_coverage_curve(conf, correct01)
    index = int(np.argmin(np.abs(coverage - target_cov)))
    return float(risk[index])


def auroc_correctness(conf: np.ndarray, correct01: np.ndarray) -> float:
    if len(np.unique(correct01)) < 2:
        return float("nan")
    order = np.argsort(conf)
    y_sorted = correct01[order].astype(np.int32)
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y_sorted) + 1)
    sum_pos_ranks = ranks[y_sorted == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def selective_metrics(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, include_curve: bool = False) -> dict:
    correct = (y_true == y_pred).astype(np.int32)
    coverage, risk = risk_coverage_curve(conf, correct)
    out = {
        "auc": auroc_correctness(conf, correct),
        "aurc": aurc(conf, correct),
        "optimal_aurc": optimal_aurc(correct),
        "eaurc": eaurc(conf, correct),
        "risk@80cov": risk_at_coverage(conf, correct, 0.80),
        "risk@90cov": risk_at_coverage(conf, correct, 0.90),
        "support": int(len(y_true)),
        "error_rate": float(1.0 - correct.mean()) if len(correct) else float("nan"),
    }
    if include_curve:
        out["coverage"] = coverage
        out["risk"] = risk
    return out


def classwise_selective_metrics(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        if mask.sum() < 5:
            out[int(cls)] = {
                "auc": float("nan"),
                "aurc": float("nan"),
                "optimal_aurc": float("nan"),
                "eaurc": float("nan"),
                "risk@80cov": float("nan"),
                "risk@90cov": float("nan"),
                "support": int(mask.sum()),
                "error_rate": float("nan"),
            }
            continue
        out[int(cls)] = selective_metrics(conf[mask], y_true[mask], y_pred[mask])
    return out
