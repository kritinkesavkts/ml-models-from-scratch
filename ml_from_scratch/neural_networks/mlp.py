from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.neural_networks.activations import relu, relu_derivative, sigmoid
from ml_from_scratch.neural_networks.losses import binary_cross_entropy


@dataclass
class MLPBinaryClassifier:
    """Multilayer perceptron for binary classification.

    This implementation uses fully connected layers, ReLU hidden activations,
    a sigmoid output layer, binary cross-entropy loss, and batch gradient
    descent with backpropagation.
    """

    hidden_layer_sizes: tuple[int, ...] = (8,)
    learning_rate: float = 0.1
    n_iterations: int = 2000
    random_state: int | None = None
    threshold: float = 0.5
    weights_: list[np.ndarray] = field(default_factory=list, init=False)
    biases_: list[np.ndarray] = field(default_factory=list, init=False)
    losses_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBinaryClassifier":
        X = self._validate_X(X)
        y = self._validate_y(y, X.shape[0]).reshape(-1, 1)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")
        if any(size <= 0 for size in self.hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive integers.")

        self._initialize_parameters(n_features=X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iterations):
            activations, pre_activations = self._forward(X)
            y_prob = activations[-1]
            self.losses_.append(binary_cross_entropy(y, y_prob))
            grad_w, grad_b = self._backward(y, activations, pre_activations)
            self._apply_gradients(grad_w, grad_b)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.weights_:
            raise ValueError("Model must be fitted before calling predict_proba.")
        X = self._validate_X(X)
        activations, _ = self._forward(X)
        return activations[-1].reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def _initialize_parameters(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        layer_sizes = (n_features, *self.hidden_layer_sizes, 1)
        self.weights_ = []
        self.biases_ = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(2.0 / fan_in)
            self.weights_.append(rng.normal(0.0, limit, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros((1, fan_out), dtype=float))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations = []

        for layer_idx, (weights, bias) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ weights + bias
            pre_activations.append(z)
            if layer_idx == len(self.weights_) - 1:
                activation = sigmoid(z)
            else:
                activation = relu(z)
            activations.append(activation)

        return activations, pre_activations

    def _backward(
        self,
        y: np.ndarray,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n_samples = y.shape[0]
        grad_w: list[np.ndarray] = [np.empty_like(w) for w in self.weights_]
        grad_b: list[np.ndarray] = [np.empty_like(b) for b in self.biases_]

        delta = activations[-1] - y
        for layer_idx in reversed(range(len(self.weights_))):
            grad_w[layer_idx] = activations[layer_idx].T @ delta / n_samples
            grad_b[layer_idx] = np.mean(delta, axis=0, keepdims=True)
            if layer_idx > 0:
                delta = (delta @ self.weights_[layer_idx].T) * relu_derivative(
                    pre_activations[layer_idx - 1]
                )

        return grad_w, grad_b

    def _apply_gradients(
        self, grad_w: list[np.ndarray], grad_b: list[np.ndarray]
    ) -> None:
        for layer_idx in range(len(self.weights_)):
            self.weights_[layer_idx] -= self.learning_rate * grad_w[layer_idx]
            self.biases_[layer_idx] -= self.learning_rate * grad_b[layer_idx]

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        return X

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=int).reshape(-1)
        if len(y) != n_samples:
            raise ValueError("y must contain one label per input sample.")
        unique = set(np.unique(y).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("MLPBinaryClassifier supports binary labels 0 and 1.")
        return y
