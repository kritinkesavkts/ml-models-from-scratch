from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.transformers import (
    MultiHeadAttention,
    ScaledDotProductAttention,
    TransformerEncoderBlock,
    sinusoidal_positional_encoding,
)


def test_scaled_dot_product_attention_shape_and_weight_sums() -> None:
    rng = np.random.default_rng(501)
    X = rng.normal(size=(3, 4, 6))
    attention = ScaledDotProductAttention()

    output = attention.forward(X, X, X)

    assert output.shape == X.shape
    assert np.allclose(attention.last_attention_weights_.sum(axis=-1), 1.0)


def test_scaled_dot_product_attention_respects_mask() -> None:
    query = np.array([[[1.0, 0.0]]])
    key = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    value = np.array([[[10.0, 0.0], [0.0, 20.0]]])
    mask = np.array([[True, False]])
    attention = ScaledDotProductAttention()

    output = attention.forward(query, key, value, mask=mask)

    assert np.allclose(output, np.array([[[10.0, 0.0]]]))
    assert attention.last_attention_weights_[0, 0, 1] == pytest.approx(0.0)


def test_multi_head_attention_preserves_model_dimension() -> None:
    rng = np.random.default_rng(502)
    X = rng.normal(size=(2, 5, 8))
    attention = MultiHeadAttention(d_model=8, n_heads=2, random_state=503)

    output = attention.forward(X)

    assert output.shape == X.shape
    assert attention.last_attention_weights_.shape == (2, 2, 5, 5)
    assert np.allclose(attention.last_attention_weights_.sum(axis=-1), 1.0)


def test_multi_head_attention_rejects_bad_head_count() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=7, n_heads=2)


def test_sinusoidal_positional_encoding_shape_and_first_position() -> None:
    encoding = sinusoidal_positional_encoding(sequence_length=4, d_model=6)

    assert encoding.shape == (4, 6)
    assert np.allclose(encoding[0, 0::2], 0.0)
    assert np.allclose(encoding[0, 1::2], 1.0)


def test_transformer_encoder_block_preserves_shape_and_normalizes() -> None:
    rng = np.random.default_rng(504)
    X = rng.normal(size=(2, 4, 8))
    encoder = TransformerEncoderBlock(d_model=8, n_heads=2, d_ff=16, random_state=505)

    output = encoder.forward(X)

    assert output.shape == X.shape
    assert np.allclose(output.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(output.std(axis=-1), 1.0, atol=1e-4)


def test_transformer_encoder_rejects_bad_input_shape() -> None:
    encoder = TransformerEncoderBlock(d_model=8, n_heads=2, d_ff=16)

    with pytest.raises(ValueError):
        encoder.forward(np.zeros((2, 4, 7)))
