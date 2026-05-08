from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.neural_networks.activations import relu
from ml_from_scratch.transformers.attention import MultiHeadAttention


@dataclass
class TransformerEncoderBlock:
    """Transformer encoder block with self-attention, feed-forward network, and layer norm."""

    d_model: int
    n_heads: int
    d_ff: int
    random_state: int | None = None
    attention_: MultiHeadAttention = field(init=False)
    W_1: np.ndarray = field(init=False)
    b_1: np.ndarray = field(init=False)
    W_2: np.ndarray = field(init=False)
    b_2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        rng = np.random.default_rng(self.random_state)
        self.attention_ = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        self.W_1 = rng.normal(0.0, np.sqrt(1.0 / self.d_model), size=(self.d_model, self.d_ff))
        self.b_1 = np.zeros((1, 1, self.d_ff))
        self.W_2 = rng.normal(0.0, np.sqrt(1.0 / self.d_ff), size=(self.d_ff, self.d_model))
        self.b_2 = np.zeros((1, 1, self.d_model))

    def forward(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[-1] != self.d_model:
            raise ValueError(f"X must have shape (batch, sequence_length, {self.d_model}).")

        attention_output = self.attention_.forward(X, mask=mask)
        X = _layer_norm(X + attention_output)
        feed_forward = relu(X @ self.W_1 + self.b_1) @ self.W_2 + self.b_2
        return _layer_norm(X + feed_forward)


def _layer_norm(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(variance + eps)
