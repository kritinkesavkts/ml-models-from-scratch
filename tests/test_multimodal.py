from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.multimodal import CLIPStyleModel


def _make_toy_pairs() -> tuple[np.ndarray, np.ndarray]:
    images = np.zeros((3, 4, 4))
    images[0, :2, :2] = 1.0
    images[1, :2, 2:] = 1.0
    images[2, 2:, :] = 1.0
    token_ids = np.array([[1, 2], [3, 4], [5, 6]])
    return images, token_ids


def _configure_aligned_model(model: CLIPStyleModel) -> None:
    model.image_projection_[:] = 0.0
    model.token_embeddings_[:] = 0.0
    model.text_projection_[:] = np.eye(model.embed_dim)

    model.image_projection_[0, 0] = 1.0
    model.image_projection_[2, 1] = 1.0
    model.image_projection_[8, 2] = 1.0

    model.token_embeddings_[1, 0] = 1.0
    model.token_embeddings_[2, 0] = 1.0
    model.token_embeddings_[3, 1] = 1.0
    model.token_embeddings_[4, 1] = 1.0
    model.token_embeddings_[5, 2] = 1.0
    model.token_embeddings_[6, 2] = 1.0


def test_clip_style_model_returns_normalized_embeddings() -> None:
    images, token_ids = _make_toy_pairs()
    model = CLIPStyleModel(image_shape=(4, 4), vocab_size=8, embed_dim=6, random_state=601)

    image_embeddings = model.encode_image(images)
    text_embeddings = model.encode_text(token_ids)

    assert image_embeddings.shape == (3, 6)
    assert text_embeddings.shape == (3, 6)
    assert np.allclose(np.linalg.norm(image_embeddings, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(text_embeddings, axis=1), 1.0)


def test_clip_style_model_retrieves_aligned_pairs() -> None:
    images, token_ids = _make_toy_pairs()
    model = CLIPStyleModel(image_shape=(4, 4), vocab_size=8, embed_dim=6, temperature=0.1)
    _configure_aligned_model(model)

    similarity = model.similarity(images, token_ids)

    assert similarity.shape == (3, 3)
    assert model.retrieve_text(images, token_ids).tolist() == [0, 1, 2]
    assert model.retrieve_image(images, token_ids).tolist() == [0, 1, 2]
    assert model.contrastive_loss(images, token_ids) < 0.01


def test_clip_style_model_logits_scale_similarity() -> None:
    images, token_ids = _make_toy_pairs()
    model = CLIPStyleModel(image_shape=(4, 4), vocab_size=8, embed_dim=6, temperature=0.5)

    assert np.allclose(model.logits(images, token_ids), model.similarity(images, token_ids) / 0.5)


def test_clip_style_model_supports_attention_mask() -> None:
    images, token_ids = _make_toy_pairs()
    token_ids = np.column_stack([token_ids, np.zeros(3, dtype=int)])
    mask = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    model = CLIPStyleModel(image_shape=(4, 4), vocab_size=8, embed_dim=6, random_state=602)

    masked = model.encode_text(token_ids, attention_mask=mask)
    unmasked = model.encode_text(token_ids[:, :2])

    assert np.allclose(masked, unmasked)


def test_clip_style_model_rejects_invalid_inputs() -> None:
    model = CLIPStyleModel(image_shape=(4, 4), vocab_size=8, embed_dim=6)

    with pytest.raises(ValueError):
        model.encode_image(np.zeros((2, 5, 4)))
    with pytest.raises(ValueError):
        model.encode_text(np.array([[1, 8]]))
    with pytest.raises(ValueError):
        model.encode_text(np.array([[1, 2]]), attention_mask=np.ones((1, 3)))
