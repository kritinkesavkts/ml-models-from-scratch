from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.transformers.encoder import TransformerEncoderBlock
from ml_from_scratch.transformers.positional_encoding import sinusoidal_positional_encoding


@dataclass
class BERTStyleEncoder:
    """Small BERT-style bidirectional Transformer encoder.

    The model combines token, segment, and positional embeddings, passes them
    through Transformer encoder blocks, and projects every token position back
    to vocabulary logits for masked language modeling.
    """

    vocab_size: int
    max_sequence_length: int
    d_model: int = 16
    n_heads: int = 2
    d_ff: int = 32
    n_layers: int = 2
    n_segments: int = 2
    random_state: int | None = None
    token_embeddings_: np.ndarray = field(init=False)
    segment_embeddings_: np.ndarray = field(init=False)
    output_projection_: np.ndarray = field(init=False)
    blocks_: list[TransformerEncoderBlock] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        if self.n_segments <= 0:
            raise ValueError("n_segments must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        rng = np.random.default_rng(self.random_state)
        self.token_embeddings_ = rng.normal(
            0.0, np.sqrt(1.0 / self.d_model), size=(self.vocab_size, self.d_model)
        )
        self.segment_embeddings_ = rng.normal(
            0.0, np.sqrt(1.0 / self.d_model), size=(self.n_segments, self.d_model)
        )
        self.output_projection_ = rng.normal(
            0.0, np.sqrt(1.0 / self.d_model), size=(self.d_model, self.vocab_size)
        )
        self.blocks_ = [
            TransformerEncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            for _ in range(self.n_layers)
        ]

    def embed(
        self, token_ids: np.ndarray, segment_ids: np.ndarray | None = None
    ) -> np.ndarray:
        token_ids = self._validate_token_ids(token_ids)
        segment_ids = self._validate_segment_ids(segment_ids, token_ids.shape)
        sequence_length = token_ids.shape[1]

        token_vectors = self.token_embeddings_[token_ids]
        segment_vectors = self.segment_embeddings_[segment_ids]
        positions = sinusoidal_positional_encoding(sequence_length, self.d_model)
        return token_vectors + segment_vectors + positions[None, :, :]

    def forward(
        self, token_ids: np.ndarray, segment_ids: np.ndarray | None = None
    ) -> np.ndarray:
        hidden = self.embed(token_ids, segment_ids)
        for block in self.blocks_:
            hidden = block.forward(hidden)
        return hidden

    def mlm_logits(
        self, token_ids: np.ndarray, segment_ids: np.ndarray | None = None
    ) -> np.ndarray:
        hidden = self.forward(token_ids, segment_ids)
        return hidden @ self.output_projection_

    def predict_masked(
        self,
        token_ids: np.ndarray,
        mask_positions: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        logits = self.mlm_logits(token_ids, segment_ids)
        token_ids = self._validate_token_ids(token_ids)
        mask_positions = np.asarray(mask_positions, dtype=int).reshape(-1)
        if len(mask_positions) != token_ids.shape[0]:
            raise ValueError("mask_positions must contain one position per sequence.")
        if np.any(mask_positions < 0) or np.any(mask_positions >= token_ids.shape[1]):
            raise ValueError("mask_positions contain invalid sequence positions.")
        batch_indices = np.arange(token_ids.shape[0])
        return np.argmax(logits[batch_indices, mask_positions, :], axis=1)

    def masked_language_modeling_loss(
        self,
        token_ids: np.ndarray,
        mask_positions: np.ndarray,
        target_token_ids: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> float:
        logits = self.mlm_logits(token_ids, segment_ids)
        token_ids = self._validate_token_ids(token_ids)
        mask_positions = np.asarray(mask_positions, dtype=int).reshape(-1)
        target_token_ids = np.asarray(target_token_ids, dtype=int).reshape(-1)
        if len(mask_positions) != token_ids.shape[0]:
            raise ValueError("mask_positions must contain one position per sequence.")
        if len(target_token_ids) != token_ids.shape[0]:
            raise ValueError("target_token_ids must contain one target per sequence.")
        if np.any(target_token_ids < 0) or np.any(target_token_ids >= self.vocab_size):
            raise ValueError("target_token_ids contain values outside the vocabulary.")

        batch_indices = np.arange(token_ids.shape[0])
        masked_logits = logits[batch_indices, mask_positions, :]
        return _cross_entropy_from_logits(masked_logits, target_token_ids)

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

    def _validate_segment_ids(
        self, segment_ids: np.ndarray | None, expected_shape: tuple[int, int]
    ) -> np.ndarray:
        if segment_ids is None:
            return np.zeros(expected_shape, dtype=int)
        segment_ids = np.asarray(segment_ids, dtype=int)
        if segment_ids.shape != expected_shape:
            raise ValueError("segment_ids must match token_ids shape.")
        if np.any(segment_ids < 0) or np.any(segment_ids >= self.n_segments):
            raise ValueError("segment_ids contain values outside the segment range.")
        return segment_ids


def _cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / np.sum(exp, axis=1, keepdims=True)
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    return float(np.mean(losses))
