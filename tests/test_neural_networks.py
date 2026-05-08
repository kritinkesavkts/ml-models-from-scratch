from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.neural_networks import MLPBinaryClassifier
from ml_from_scratch.neural_networks.activations import relu, sigmoid
from ml_from_scratch.neural_networks.losses import binary_cross_entropy
from ml_from_scratch.utils import accuracy_score


def _make_xor_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(201)
    base_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    base_y = np.array([0, 1, 1, 0])
    X = np.repeat(base_X, 70, axis=0)
    y = np.repeat(base_y, 70)
    X = X + rng.normal(scale=0.06, size=X.shape)
    return X, y


def test_sigmoid_and_relu_shapes_and_ranges() -> None:
    z = np.array([[-2.0, 0.0, 2.0]])

    assert sigmoid(z).shape == z.shape
    assert np.all((sigmoid(z) > 0.0) & (sigmoid(z) < 1.0))
    assert relu(z).tolist() == [[0.0, 0.0, 2.0]]


def test_binary_cross_entropy_is_lower_for_better_probabilities() -> None:
    y = np.array([0, 1, 1, 0])
    good = np.array([[0.05], [0.95], [0.9], [0.1]])
    bad = np.array([[0.95], [0.05], [0.1], [0.9]])

    assert binary_cross_entropy(y, good) < binary_cross_entropy(y, bad)


def test_mlp_binary_classifier_learns_xor_boundary() -> None:
    X, y = _make_xor_data()
    model = MLPBinaryClassifier(
        hidden_layer_sizes=(8, 4),
        learning_rate=0.2,
        n_iterations=2500,
        random_state=202,
    )

    model.fit(X, y)

    assert accuracy_score(y, model.predict(X)) >= 0.95
    assert model.losses_[0] > model.losses_[-1]


def test_mlp_requires_fit_before_prediction() -> None:
    with pytest.raises(ValueError):
        MLPBinaryClassifier().predict(np.array([[0.0, 1.0]]))


def test_mlp_rejects_non_binary_labels() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        MLPBinaryClassifier().fit(X, y)
