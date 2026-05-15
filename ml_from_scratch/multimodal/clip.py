from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CLIPStyleModel:
    """Small CLIP-style image-text embedding model.

    This model projects images and tokenized text into a shared embedding
    space. Similarity is computed with normalized dot products, which is the
    core retrieval idea behind CLIP-style systems.
    """

    image_shape: tuple[int, int]
    vocab_size: int
    embed_dim: int = 16
    random_state: int | None = None
    temperature: float = 0.07
    image_projection_: np.ndarray = field(init=False)
    token_embeddings_: np.ndarray = field(init=False)
    text_projection_: np.ndarray = field(init=False)
    losses_: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must be a positive (height, width) tuple.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

        rng = np.random.default_rng(self.random_state)
        image_dim = self.image_shape[0] * self.image_shape[1]
        self.image_projection_ = rng.normal(
            0.0, np.sqrt(1.0 / image_dim), size=(image_dim, self.embed_dim)
        )
        self.token_embeddings_ = rng.normal(
            0.0, np.sqrt(1.0 / self.embed_dim), size=(self.vocab_size, self.embed_dim)
        )
        self.text_projection_ = rng.normal(
            0.0, np.sqrt(1.0 / self.embed_dim), size=(self.embed_dim, self.embed_dim)
        )

    def encode_image(self, images: np.ndarray) -> np.ndarray:
        images = self._validate_images(images)
        flattened = images.reshape(images.shape[0], -1)
        return _l2_normalize(flattened @ self.image_projection_)

    def encode_text(self, token_ids: np.ndarray, attention_mask: np.ndarray | None = None) -> np.ndarray:
        token_ids = self._validate_token_ids(token_ids)
        token_vectors = self.token_embeddings_[token_ids]

        if attention_mask is None:
            mask = np.ones(token_ids.shape, dtype=float)
        else:
            mask = np.asarray(attention_mask, dtype=float)
            if mask.shape != token_ids.shape:
                raise ValueError("attention_mask must match token_ids shape.")

        lengths = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        pooled = (token_vectors * mask[:, :, None]).sum(axis=1) / lengths
        return _l2_normalize(pooled @ self.text_projection_)

    def similarity(self, images: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(token_ids)
        return image_embeddings @ text_embeddings.T

    def logits(self, images: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        return self.similarity(images, token_ids) / self.temperature

    def contrastive_loss(self, images: np.ndarray, token_ids: np.ndarray) -> float:
        logits = self.logits(images, token_ids)
        if logits.shape[0] != logits.shape[1]:
            raise ValueError("contrastive_loss expects the same number of images and texts.")

        labels = np.arange(logits.shape[0])
        image_to_text = _cross_entropy_from_logits(logits, labels)
        text_to_image = _cross_entropy_from_logits(logits.T, labels)
        return float((image_to_text + text_to_image) / 2.0)

    def retrieve_text(self, images: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        return np.argmax(self.similarity(images, token_ids), axis=1)

    def retrieve_image(self, images: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        return np.argmax(self.similarity(images, token_ids), axis=0)

    def fit(
        self,
        images: np.ndarray,
        token_ids: np.ndarray,
        learning_rate: float = 0.05,
        n_iterations: int = 500,
    ) -> "CLIPStyleModel":
        """Train image/text projections with symmetric contrastive loss."""
        images = self._validate_images(images)
        token_ids = self._validate_token_ids(token_ids)
        if images.shape[0] != token_ids.shape[0]:
            raise ValueError("fit expects the same number of images and texts.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")

        self.losses_ = []
        flattened_images = images.reshape(images.shape[0], -1)

        for _ in range(n_iterations):
            cache = self._forward_training(flattened_images, token_ids)
            self.losses_.append(cache["loss"])
            gradients = self._backward_training(flattened_images, token_ids, cache)
            self.image_projection_ -= learning_rate * gradients["image_projection"]
            self.text_projection_ -= learning_rate * gradients["text_projection"]
            self.token_embeddings_ -= learning_rate * gradients["token_embeddings"]

        return self

    def _validate_images(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=float)
        if images.ndim != 3:
            raise ValueError("images must have shape (batch, height, width).")
        if images.shape[1:] != self.image_shape:
            raise ValueError(f"images must have shape (batch, {self.image_shape[0]}, {self.image_shape[1]}).")
        return images

    def _validate_token_ids(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=int)
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence_length).")
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError("token_ids contain values outside the vocabulary.")
        return token_ids

    def _forward_training(
        self, flattened_images: np.ndarray, token_ids: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        image_raw = flattened_images @ self.image_projection_
        image_embeddings = _l2_normalize(image_raw)

        token_vectors = self.token_embeddings_[token_ids]
        text_pooled = token_vectors.mean(axis=1)
        text_raw = text_pooled @ self.text_projection_
        text_embeddings = _l2_normalize(text_raw)

        logits = image_embeddings @ text_embeddings.T / self.temperature
        labels = np.arange(logits.shape[0])
        image_loss, image_probs = _cross_entropy_with_probs(logits, labels)
        text_loss, text_probs = _cross_entropy_with_probs(logits.T, labels)
        loss = float((image_loss + text_loss) / 2.0)

        return {
            "image_raw": image_raw,
            "image_embeddings": image_embeddings,
            "text_pooled": text_pooled,
            "text_raw": text_raw,
            "text_embeddings": text_embeddings,
            "image_probs": image_probs,
            "text_probs": text_probs,
            "loss": loss,
        }

    def _backward_training(
        self,
        flattened_images: np.ndarray,
        token_ids: np.ndarray,
        cache: dict[str, np.ndarray | float],
    ) -> dict[str, np.ndarray]:
        n_samples = token_ids.shape[0]
        labels = np.arange(n_samples)

        d_logits_image = cache["image_probs"].copy()
        d_logits_image[np.arange(n_samples), labels] -= 1.0
        d_logits_image /= n_samples

        d_logits_text = cache["text_probs"].copy()
        d_logits_text[np.arange(n_samples), labels] -= 1.0
        d_logits_text /= n_samples

        d_similarity = (d_logits_image + d_logits_text.T) / (2.0 * self.temperature)
        d_image_embeddings = d_similarity @ cache["text_embeddings"]
        d_text_embeddings = d_similarity.T @ cache["image_embeddings"]

        d_image_raw = _l2_normalize_backward(d_image_embeddings, cache["image_raw"])
        d_text_raw = _l2_normalize_backward(d_text_embeddings, cache["text_raw"])

        d_image_projection = flattened_images.T @ d_image_raw
        d_text_projection = cache["text_pooled"].T @ d_text_raw
        d_text_pooled = d_text_raw @ self.text_projection_.T

        d_token_embeddings = np.zeros_like(self.token_embeddings_)
        token_weight = 1.0 / token_ids.shape[1]
        for row, sequence in enumerate(token_ids):
            for token_id in sequence:
                d_token_embeddings[token_id] += d_text_pooled[row] * token_weight

        return {
            "image_projection": np.clip(d_image_projection, -5.0, 5.0),
            "text_projection": np.clip(d_text_projection, -5.0, 5.0),
            "token_embeddings": np.clip(d_token_embeddings, -5.0, 5.0),
        }


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / np.sum(exp, axis=1, keepdims=True)
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    return float(np.mean(losses))


def _cross_entropy_with_probs(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / np.sum(exp, axis=1, keepdims=True)
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    return float(np.mean(losses)), probabilities


def _l2_normalize_backward(d_normalized: np.ndarray, raw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), eps)
    normalized = raw / norms
    projection = np.sum(d_normalized * normalized, axis=1, keepdims=True)
    return (d_normalized - normalized * projection) / norms
