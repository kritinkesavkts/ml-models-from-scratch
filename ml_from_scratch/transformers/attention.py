from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScaledDotProductAttention:
    """Scaled dot-product attention.

    Attention compares queries with keys, normalizes the scores, then uses
    those weights to mix values.
    """

    last_attention_weights_: np.ndarray | None = field(default=None, init=False)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        query = _validate_sequence_tensor(query, "query")
        key = _validate_sequence_tensor(key, "key")
        value = _validate_sequence_tensor(value, "value")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("key and value must share batch and sequence dimensions.")
        if query.shape[0] != key.shape[0]:
            raise ValueError("query and key must share batch dimension.")

        scores = query @ np.swapaxes(key, -1, -2)
        scores = scores / np.sqrt(query.shape[-1])
        if mask is not None:
            mask = _validate_mask(mask, scores.shape)
            scores = np.where(mask, scores, -1e9)

        attention_weights = _softmax(scores, axis=-1)
        self.last_attention_weights_ = attention_weights
        return attention_weights @ value


@dataclass
class MultiHeadAttention:
    """Multi-head self-attention with learned linear projections."""

    d_model: int
    n_heads: int
    random_state: int | None = None
    W_q: np.ndarray = field(init=False)
    W_k: np.ndarray = field(init=False)
    W_v: np.ndarray = field(init=False)
    W_o: np.ndarray = field(init=False)
    last_attention_weights_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / self.d_model)
        self.W_q = rng.normal(0.0, scale, size=(self.d_model, self.d_model))
        self.W_k = rng.normal(0.0, scale, size=(self.d_model, self.d_model))
        self.W_v = rng.normal(0.0, scale, size=(self.d_model, self.d_model))
        self.W_o = rng.normal(0.0, scale, size=(self.d_model, self.d_model))

    def forward(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        X = _validate_sequence_tensor(X, "X")
        if X.shape[-1] != self.d_model:
            raise ValueError(f"X last dimension must equal d_model={self.d_model}.")

        Q = self._split_heads(X @ self.W_q)
        K = self._split_heads(X @ self.W_k)
        V = self._split_heads(X @ self.W_v)

        scores = Q @ np.swapaxes(K, -1, -2)
        scores = scores / np.sqrt(self.d_model // self.n_heads)
        if mask is not None:
            mask = self._prepare_mask(mask, scores.shape)
            scores = np.where(mask, scores, -1e9)

        weights = _softmax(scores, axis=-1)
        self.last_attention_weights_ = weights
        context = weights @ V
        combined = self._combine_heads(context)
        return combined @ self.W_o

    def _split_heads(self, X: np.ndarray) -> np.ndarray:
        batch_size, sequence_length, _ = X.shape
        head_dim = self.d_model // self.n_heads
        X = X.reshape(batch_size, sequence_length, self.n_heads, head_dim)
        return np.transpose(X, (0, 2, 1, 3))

    def _combine_heads(self, X: np.ndarray) -> np.ndarray:
        batch_size, _, sequence_length, head_dim = X.shape
        X = np.transpose(X, (0, 2, 1, 3))
        return X.reshape(batch_size, sequence_length, self.n_heads * head_dim)

    def _prepare_mask(self, mask: np.ndarray, score_shape: tuple[int, ...]) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == score_shape:
            return mask
        if mask.ndim == 3 and mask.shape == (score_shape[0], score_shape[2], score_shape[3]):
            return mask[:, None, :, :]
        if mask.ndim == 2 and mask.shape == (score_shape[2], score_shape[3]):
            return mask[None, None, :, :]
        raise ValueError("mask must broadcast to attention score shape.")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _validate_sequence_tensor(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (batch, sequence_length, features).")
    return X


def _validate_mask(mask: np.ndarray, score_shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == score_shape:
        return mask
    if mask.ndim == 2 and mask.shape == score_shape[-2:]:
        return mask[None, :, :]
    if mask.ndim == 3 and mask.shape == score_shape:
        return mask
    raise ValueError("mask must match attention scores or sequence dimensions.")
