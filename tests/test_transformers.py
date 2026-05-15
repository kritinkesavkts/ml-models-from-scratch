from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.transformers import (
    BERTStyleEncoder,
    GPTStyleDecoder,
    MultiHeadAttention,
    ScaledDotProductAttention,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
    VisionTransformerClassifier,
    causal_mask,
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


def test_causal_mask_blocks_future_positions() -> None:
    mask = causal_mask(4)

    assert mask.dtype == bool
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_transformer_decoder_block_preserves_shape_and_uses_causal_attention() -> None:
    rng = np.random.default_rng(506)
    X = rng.normal(size=(2, 5, 8))
    decoder = TransformerDecoderBlock(d_model=8, n_heads=2, d_ff=16, random_state=507)

    output = decoder.forward(X)
    weights = decoder.attention_.last_attention_weights_

    assert output.shape == X.shape
    assert weights.shape == (2, 2, 5, 5)
    assert np.allclose(weights[:, :, 0, 1:], 0.0)
    assert np.allclose(output.mean(axis=-1), 0.0, atol=1e-6)


def test_gpt_style_decoder_returns_next_token_logits() -> None:
    model = GPTStyleDecoder(
        vocab_size=10,
        max_sequence_length=6,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        random_state=508,
    )
    token_ids = np.array([[1, 2, 3], [3, 2, 1]])

    logits = model.forward(token_ids)
    next_tokens = model.predict_next(token_ids)

    assert logits.shape == (2, 3, 10)
    assert next_tokens.shape == (2,)
    assert np.all((next_tokens >= 0) & (next_tokens < 10))


def test_gpt_style_decoder_generate_extends_prompt_until_limit() -> None:
    model = GPTStyleDecoder(
        vocab_size=7,
        max_sequence_length=5,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        random_state=509,
    )
    prompt = np.array([[1, 2, 3]])

    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (1, 5)
    assert generated[:, :3].tolist() == [[1, 2, 3]]


def test_gpt_style_decoder_rejects_bad_token_ids() -> None:
    model = GPTStyleDecoder(vocab_size=5, max_sequence_length=4, d_model=8, n_heads=2)

    with pytest.raises(ValueError):
        model.forward(np.array([[1, 5]]))


def test_bert_style_encoder_returns_hidden_states_and_mlm_logits() -> None:
    model = BERTStyleEncoder(
        vocab_size=12,
        max_sequence_length=6,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        random_state=513,
    )
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    segment_ids = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])

    hidden = model.forward(token_ids, segment_ids)
    logits = model.mlm_logits(token_ids, segment_ids)

    assert hidden.shape == (2, 4, 8)
    assert logits.shape == (2, 4, 12)
    assert np.allclose(hidden.mean(axis=-1), 0.0, atol=1e-6)


def test_bert_style_encoder_predicts_masked_positions() -> None:
    model = BERTStyleEncoder(
        vocab_size=12,
        max_sequence_length=6,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        random_state=514,
    )
    token_ids = np.array([[1, 11, 3], [4, 5, 11]])
    mask_positions = np.array([1, 2])

    predictions = model.predict_masked(token_ids, mask_positions)
    loss = model.masked_language_modeling_loss(
        token_ids, mask_positions, np.array([2, 6])
    )

    assert predictions.shape == (2,)
    assert np.all((predictions >= 0) & (predictions < 12))
    assert loss > 0.0


def test_bert_style_encoder_uses_segment_embeddings() -> None:
    model = BERTStyleEncoder(
        vocab_size=12,
        max_sequence_length=6,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        random_state=515,
    )
    token_ids = np.array([[1, 2, 3]])
    segment_a = np.array([[0, 0, 0]])
    segment_b = np.array([[1, 1, 1]])

    embedded_a = model.embed(token_ids, segment_a)
    embedded_b = model.embed(token_ids, segment_b)

    assert not np.allclose(embedded_a, embedded_b)


def test_bert_style_encoder_rejects_invalid_inputs() -> None:
    model = BERTStyleEncoder(vocab_size=5, max_sequence_length=4, d_model=8, n_heads=2)

    with pytest.raises(ValueError):
        model.forward(np.array([[1, 5]]))
    with pytest.raises(ValueError):
        model.forward(np.array([[1, 2, 3, 4, 1]]))
    with pytest.raises(ValueError):
        model.forward(np.array([[1, 2]]), segment_ids=np.array([[0, 2]]))
    with pytest.raises(ValueError):
        model.predict_masked(np.array([[1, 2]]), np.array([2]))


def test_vision_transformer_extracts_non_overlapping_patches() -> None:
    image = np.arange(16).reshape(1, 4, 4)
    model = VisionTransformerClassifier(
        image_size=4,
        patch_size=2,
        n_classes=2,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        random_state=510,
    )

    patches = model.extract_patches(image)

    assert patches.shape == (1, 4, 4)
    assert patches[0, 0].tolist() == [0, 1, 4, 5]
    assert patches[0, 3].tolist() == [10, 11, 14, 15]


def test_vision_transformer_embedding_and_logits_shapes() -> None:
    rng = np.random.default_rng(511)
    images = rng.normal(size=(2, 8, 8))
    model = VisionTransformerClassifier(
        image_size=8,
        patch_size=2,
        n_classes=3,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        random_state=512,
    )

    tokens = model.embed_patches(images)
    logits = model.forward(images)
    predictions = model.predict(images)

    assert model.n_patches_ == 16
    assert tokens.shape == (2, 17, 8)
    assert logits.shape == (2, 3)
    assert predictions.shape == (2,)
    assert np.all((predictions >= 0) & (predictions < 3))


def test_vision_transformer_rejects_bad_image_shape() -> None:
    model = VisionTransformerClassifier(image_size=8, patch_size=2, n_classes=2)

    with pytest.raises(ValueError):
        model.forward(np.zeros((2, 7, 8)))


def test_vision_transformer_requires_divisible_patch_size() -> None:
    with pytest.raises(ValueError):
        VisionTransformerClassifier(image_size=7, patch_size=2, n_classes=2)
