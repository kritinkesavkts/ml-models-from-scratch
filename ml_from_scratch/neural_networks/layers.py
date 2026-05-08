from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Conv2D:
    """Naive 2D convolution layer for grayscale images.

    Input shape is `(batch, height, width)`. Output shape is
    `(batch, n_filters, out_height, out_width)`.
    """

    n_filters: int
    kernel_size: int
    random_state: int | None = None
    kernels_: np.ndarray | None = field(default=None, init=False)
    biases_: np.ndarray | None = field(default=None, init=False)

    def initialize(self) -> None:
        if self.n_filters <= 0:
            raise ValueError("n_filters must be positive.")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(2.0 / (self.kernel_size * self.kernel_size))
        self.kernels_ = rng.normal(
            0.0, scale, size=(self.n_filters, self.kernel_size, self.kernel_size)
        )
        self.biases_ = np.zeros(self.n_filters, dtype=float)

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = _validate_images(X)
        if self.kernels_ is None or self.biases_ is None:
            self.initialize()

        batch_size, height, width = X.shape
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        if out_height <= 0 or out_width <= 0:
            raise ValueError("kernel_size cannot exceed image height or width.")

        output = np.zeros((batch_size, self.n_filters, out_height, out_width))
        for row in range(out_height):
            for col in range(out_width):
                patch = X[:, row : row + self.kernel_size, col : col + self.kernel_size]
                output[:, :, row, col] = np.tensordot(
                    patch, self.kernels_, axes=([1, 2], [1, 2])
                ) + self.biases_
        return output


@dataclass
class MaxPool2D:
    """Max pooling layer for 4D convolution outputs."""

    pool_size: int = 2
    stride: int = 2

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 4:
            raise ValueError("X must have shape (batch, channels, height, width).")
        if self.pool_size <= 0 or self.stride <= 0:
            raise ValueError("pool_size and stride must be positive.")

        batch_size, channels, height, width = X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        if out_height <= 0 or out_width <= 0:
            raise ValueError("pool_size cannot exceed input height or width.")

        output = np.zeros((batch_size, channels, out_height, out_width))
        for row in range(out_height):
            for col in range(out_width):
                row_start = row * self.stride
                col_start = col * self.stride
                patch = X[
                    :,
                    :,
                    row_start : row_start + self.pool_size,
                    col_start : col_start + self.pool_size,
                ]
                output[:, :, row, col] = np.max(patch, axis=(2, 3))
        return output


class Flatten:
    """Flatten all dimensions except the batch dimension."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim < 2:
            raise ValueError("X must include a batch dimension.")
        return X.reshape(X.shape[0], -1)


def _validate_images(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must have shape (batch, height, width).")
    return X
