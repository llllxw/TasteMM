import numpy as np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)


def conf_margin_from_logits(logits: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    part = np.partition(-probs, 1, axis=1)
    top1 = -part[:, 0]
    top2 = -part[:, 1]
    return top1 - top2


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / (float(temperature) + 1e-12)
