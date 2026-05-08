from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.transformers.encoder import TransformerEncoderBlock
from ml_from_scratch.transformers.positional_encoding import sinusoidal_positional_encoding


@dataclass
class VisionTransformerClassifier:
    """Small Vision Transformer-style image classifier.

    Images are split into patches, each patch is projected into a token, a
    learned class token is prepended, positional encodings are added, and a
    stack of Transformer encoder blocks produces class logits.
    """

    image_size: int
    patch_size: int
    n_classes: int
    d_model: int = 16
    n_heads: int = 2
    d_ff: int = 32
    n_layers: int = 2
    random_state: int | None = None
    patch_projection_: np.ndarray = field(init=False)
    class_token_: np.ndarray = field(init=False)
    output_projection_: np.ndarray = field(init=False)
    blocks_: list[TransformerEncoderBlock] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if self.n_classes <= 1:
            raise ValueError("n_classes must be greater than 1.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        rng = np.random.default_rng(self.random_state)
        patch_dim = self.patch_size * self.patch_size
        self.patch_projection_ = rng.normal(
            0.0, np.sqrt(1.0 / patch_dim), size=(patch_dim, self.d_model)
        )
        self.class_token_ = rng.normal(0.0, 0.02, size=(1, 1, self.d_model))
        self.output_projection_ = rng.normal(
            0.0, np.sqrt(1.0 / self.d_model), size=(self.d_model, self.n_classes)
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

    @property
    def n_patches_(self) -> int:
        patches_per_axis = self.image_size // self.patch_size
        return patches_per_axis * patches_per_axis

    def extract_patches(self, images: np.ndarray) -> np.ndarray:
        images = self._validate_images(images)
        batch_size = images.shape[0]
        patches = []
        for row in range(0, self.image_size, self.patch_size):
            for col in range(0, self.image_size, self.patch_size):
                patch = images[:, row : row + self.patch_size, col : col + self.patch_size]
                patches.append(patch.reshape(batch_size, -1))
        return np.stack(patches, axis=1)

    def embed_patches(self, images: np.ndarray) -> np.ndarray:
        patches = self.extract_patches(images)
        patch_tokens = patches @ self.patch_projection_
        class_tokens = np.repeat(self.class_token_, repeats=patch_tokens.shape[0], axis=0)
        tokens = np.concatenate([class_tokens, patch_tokens], axis=1)
        positions = sinusoidal_positional_encoding(tokens.shape[1], self.d_model)
        return tokens + positions[None, :, :]

    def forward(self, images: np.ndarray) -> np.ndarray:
        tokens = self.embed_patches(images)
        for block in self.blocks_:
            tokens = block.forward(tokens)
        class_representation = tokens[:, 0, :]
        return class_representation @ self.output_projection_

    def predict(self, images: np.ndarray) -> np.ndarray:
        logits = self.forward(images)
        return np.argmax(logits, axis=1)

    def _validate_images(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=float)
        if images.ndim != 3:
            raise ValueError("images must have shape (batch, height, width).")
        if images.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(
                f"images must have shape (batch, {self.image_size}, {self.image_size})."
            )
        return images
