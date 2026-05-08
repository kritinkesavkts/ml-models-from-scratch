from __future__ import annotations

import numpy as np


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.mean(loss))
