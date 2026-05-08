from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.neural_networks.activations import relu
from ml_from_scratch.neural_networks.layers import Conv2D, Flatten, MaxPool2D
from ml_from_scratch.neural_networks.mlp import MLPBinaryClassifier


@dataclass
class SimpleCNNBinaryClassifier:
    """Small CNN-style binary classifier using fixed convolutional features.

    The convolution and pooling layers extract local image patterns, then an
    MLP classifier learns from the flattened feature map. This keeps the model
    compact while making each CNN operation explicit and inspectable.
    """

    n_filters: int = 4
    kernel_size: int = 3
    pool_size: int = 2
    hidden_layer_sizes: tuple[int, ...] = (8,)
    learning_rate: float = 0.1
    n_iterations: int = 1500
    random_state: int | None = None
    conv_: Conv2D | None = field(default=None, init=False)
    pool_: MaxPool2D | None = field(default=None, init=False)
    flatten_: Flatten = field(default_factory=Flatten, init=False)
    classifier_: MLPBinaryClassifier | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleCNNBinaryClassifier":
        X = self._validate_images(X)
        rng = np.random.default_rng(self.random_state)
        self.conv_ = Conv2D(
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        self.pool_ = MaxPool2D(pool_size=self.pool_size, stride=self.pool_size)
        features = self._extract_features(X)
        self.classifier_ = MLPBinaryClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        self.classifier_.fit(features, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X = self._validate_images(X)
        return self.classifier_.predict_proba(self._extract_features(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X = self._validate_images(X)
        return self.classifier_.predict(self._extract_features(X))

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        if self.conv_ is None or self.pool_ is None:
            raise ValueError("Convolution and pooling layers are not initialized.")
        conv_out = relu(self.conv_.forward(X))
        pooled = self.pool_.forward(conv_out)
        return self.flatten_.forward(pooled)

    @staticmethod
    def _validate_images(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, height, width).")
        return X
