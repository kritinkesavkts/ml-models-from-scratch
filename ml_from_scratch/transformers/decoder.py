from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.neural_networks.activations import relu
from ml_from_scratch.transformers.attention import MultiHeadAttention
from ml_from_scratch.transformers.positional_encoding import sinusoidal_positional_encoding


def causal_mask(sequence_length: int) -> np.ndarray:
    """Return a lower-triangular mask that blocks future tokens."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    return np.tril(np.ones((sequence_length, sequence_length), dtype=bool))


@dataclass
class TransformerDecoderBlock:
    """Decoder-only Transformer block with causal self-attention."""

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

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[-1] != self.d_model:
            raise ValueError(f"X must have shape (batch, sequence_length, {self.d_model}).")

        mask = causal_mask(X.shape[1])
        attention_output = self.attention_.forward(X, mask=mask)
        X = _layer_norm(X + attention_output)
        feed_forward = relu(X @ self.W_1 + self.b_1) @ self.W_2 + self.b_2
        return _layer_norm(X + feed_forward)


@dataclass
class GPTStyleDecoder:
    """Tiny GPT-style decoder stack for next-token logits and greedy generation."""

    vocab_size: int
    max_sequence_length: int
    d_model: int = 16
    n_heads: int = 2
    d_ff: int = 32
    n_layers: int = 2
    random_state: int | None = None
    token_embeddings_: np.ndarray = field(init=False)
    output_projection_: np.ndarray = field(init=False)
    blocks_: list[TransformerDecoderBlock] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        rng = np.random.default_rng(self.random_state)
        self.token_embeddings_ = rng.normal(
            0.0, np.sqrt(1.0 / self.d_model), size=(self.vocab_size, self.d_model)
        )
        self.output_projection_ = rng.normal(
            0.0, np.sqrt(1.0 / self.d_model), size=(self.d_model, self.vocab_size)
        )
        self.blocks_ = [
            TransformerDecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            for _ in range(self.n_layers)
        ]

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = self._validate_token_ids(token_ids)
        _, sequence_length = token_ids.shape
        X = self.token_embeddings_[token_ids]
        positions = sinusoidal_positional_encoding(sequence_length, self.d_model)
        X = X + positions[None, :, :]

        for block in self.blocks_:
            X = block.forward(X)

        return X @ self.output_projection_

    def predict_next(self, token_ids: np.ndarray) -> np.ndarray:
        logits = self.forward(token_ids)
        return np.argmax(logits[:, -1, :], axis=-1)

    def generate(self, prompt: np.ndarray, max_new_tokens: int) -> np.ndarray:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        tokens = self._validate_token_ids(prompt)

        for _ in range(max_new_tokens):
            if tokens.shape[1] >= self.max_sequence_length:
                break
            next_token = self.predict_next(tokens)[:, None]
            tokens = np.concatenate([tokens, next_token], axis=1)
        return tokens

    def _validate_token_ids(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=int)
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence_length).")
        if token_ids.shape[1] > self.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length.")
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError("token_ids contain values outside the vocabulary.")
        return token_ids


def _layer_norm(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(variance + eps)
