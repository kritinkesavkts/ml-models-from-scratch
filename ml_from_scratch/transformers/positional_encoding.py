from __future__ import annotations

import numpy as np


def sinusoidal_positional_encoding(sequence_length: int, d_model: int) -> np.ndarray:
    """Return classic Transformer sinusoidal positional encodings."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if d_model <= 0:
        raise ValueError("d_model must be positive.")

    positions = np.arange(sequence_length)[:, None]
    dimensions = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (dimensions // 2)) / d_model)
    angles = positions * angle_rates

    encoding = np.zeros((sequence_length, d_model), dtype=float)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding
