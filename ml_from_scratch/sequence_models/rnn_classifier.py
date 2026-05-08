from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.neural_networks.activations import sigmoid
from ml_from_scratch.neural_networks.losses import binary_cross_entropy


@dataclass
class SimpleRNNBinaryClassifier:
    """Trainable vanilla RNN for binary sequence classification.

    The model reads a sequence one step at a time, uses the final hidden state,
    and predicts a binary label with a sigmoid output layer.
    """

    hidden_size: int = 8
    learning_rate: float = 0.1
    n_iterations: int = 1000
    random_state: int | None = None
    threshold: float = 0.5
    W_xh: np.ndarray | None = field(default=None, init=False)
    W_hh: np.ndarray | None = field(default=None, init=False)
    b_h: np.ndarray | None = field(default=None, init=False)
    W_hy: np.ndarray | None = field(default=None, init=False)
    b_y: float = field(default=0.0, init=False)
    losses_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRNNBinaryClassifier":
        X = self._validate_sequences(X)
        y = self._validate_y(y, X.shape[0]).reshape(-1, 1)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self._initialize(input_size=X.shape[2])
        self.losses_ = []

        for _ in range(self.n_iterations):
            cache = self._forward(X)
            y_prob = cache["y_prob"]
            self.losses_.append(binary_cross_entropy(y, y_prob))
            grads = self._backward(X, y, cache)
            self._apply_gradients(grads)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W_xh is None:
            raise ValueError("Model must be fitted before prediction.")
        X = self._validate_sequences(X)
        return self._forward(X)["y_prob"].reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def _initialize(self, input_size: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.W_xh = rng.normal(0.0, np.sqrt(1.0 / input_size), size=(input_size, self.hidden_size))
        self.W_hh = rng.normal(0.0, np.sqrt(1.0 / self.hidden_size), size=(self.hidden_size, self.hidden_size))
        self.b_h = np.zeros((1, self.hidden_size))
        self.W_hy = rng.normal(0.0, np.sqrt(1.0 / self.hidden_size), size=(self.hidden_size, 1))
        self.b_y = 0.0

    def _forward(self, X: np.ndarray) -> dict[str, np.ndarray | list[np.ndarray]]:
        batch_size, sequence_length, _ = X.shape
        hidden_states = [np.zeros((batch_size, self.hidden_size))]
        pre_activations = []

        for step in range(sequence_length):
            z_t = X[:, step, :] @ self.W_xh + hidden_states[-1] @ self.W_hh + self.b_h
            h_t = np.tanh(z_t)
            pre_activations.append(z_t)
            hidden_states.append(h_t)

        logits = hidden_states[-1] @ self.W_hy + self.b_y
        y_prob = sigmoid(logits)
        return {
            "hidden_states": hidden_states,
            "pre_activations": pre_activations,
            "y_prob": y_prob,
        }

    def _backward(
        self, X: np.ndarray, y: np.ndarray, cache: dict[str, np.ndarray | list[np.ndarray]]
    ) -> dict[str, np.ndarray | float]:
        batch_size, sequence_length, _ = X.shape
        hidden_states = cache["hidden_states"]
        pre_activations = cache["pre_activations"]
        y_prob = cache["y_prob"]

        d_logits = (y_prob - y) / batch_size
        dW_hy = hidden_states[-1].T @ d_logits
        db_y = float(np.sum(d_logits))
        dh_next = d_logits @ self.W_hy.T

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        for step in reversed(range(sequence_length)):
            dz = dh_next * (1.0 - np.tanh(pre_activations[step]) ** 2)
            dW_xh += X[:, step, :].T @ dz
            dW_hh += hidden_states[step].T @ dz
            db_h += np.sum(dz, axis=0, keepdims=True)
            dh_next = dz @ self.W_hh.T

        return {
            "W_xh": np.clip(dW_xh, -5.0, 5.0),
            "W_hh": np.clip(dW_hh, -5.0, 5.0),
            "b_h": np.clip(db_h, -5.0, 5.0),
            "W_hy": np.clip(dW_hy, -5.0, 5.0),
            "b_y": float(np.clip(db_y, -5.0, 5.0)),
        }

    def _apply_gradients(self, grads: dict[str, np.ndarray | float]) -> None:
        self.W_xh -= self.learning_rate * grads["W_xh"]
        self.W_hh -= self.learning_rate * grads["W_hh"]
        self.b_h -= self.learning_rate * grads["b_h"]
        self.W_hy -= self.learning_rate * grads["W_hy"]
        self.b_y -= self.learning_rate * grads["b_y"]

    @staticmethod
    def _validate_sequences(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, sequence_length, input_size).")
        return X

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=int).reshape(-1)
        if len(y) != n_samples:
            raise ValueError("y must contain one label per input sequence.")
        unique = set(np.unique(y).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("SimpleRNNBinaryClassifier supports labels 0 and 1.")
        return y
