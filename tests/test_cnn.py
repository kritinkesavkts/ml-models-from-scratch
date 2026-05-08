from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.neural_networks import Conv2D, Flatten, MaxPool2D, SimpleCNNBinaryClassifier
from ml_from_scratch.utils import accuracy_score


def _make_bar_images(samples_per_class: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(301)
    images = []
    labels = []

    for _ in range(samples_per_class):
        vertical = rng.normal(scale=0.06, size=(8, 8))
        vertical[:, 3:5] += 1.0
        images.append(vertical)
        labels.append(1)

        horizontal = rng.normal(scale=0.06, size=(8, 8))
        horizontal[3:5, :] += 1.0
        images.append(horizontal)
        labels.append(0)

    return np.array(images), np.array(labels)


def test_conv2d_forward_shape_and_values_with_known_kernel() -> None:
    layer = Conv2D(n_filters=1, kernel_size=2)
    layer.kernels_ = np.ones((1, 2, 2))
    layer.biases_ = np.array([0.5])
    X = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

    output = layer.forward(X)

    assert output.shape == (1, 1, 2, 2)
    assert output[0, 0].tolist() == [[12.5, 16.5], [24.5, 28.5]]


def test_max_pool2d_forward_shape_and_values() -> None:
    layer = MaxPool2D(pool_size=2, stride=2)
    X = np.array([[[[1.0, 3.0, 2.0, 4.0], [5.0, 6.0, 7.0, 8.0], [2.0, 1.0, 9.0, 0.0], [3.0, 4.0, 5.0, 6.0]]]])

    output = layer.forward(X)

    assert output.shape == (1, 1, 2, 2)
    assert output[0, 0].tolist() == [[6.0, 8.0], [4.0, 9.0]]


def test_flatten_keeps_batch_dimension() -> None:
    X = np.zeros((3, 2, 4, 5))

    assert Flatten().forward(X).shape == (3, 40)


def test_simple_cnn_binary_classifier_learns_bar_images() -> None:
    X, y = _make_bar_images()
    model = SimpleCNNBinaryClassifier(
        n_filters=6,
        kernel_size=3,
        pool_size=2,
        hidden_layer_sizes=(10,),
        learning_rate=0.15,
        n_iterations=1200,
        random_state=302,
    )

    model.fit(X, y)

    assert accuracy_score(y, model.predict(X)) >= 0.95


def test_cnn_requires_fit_before_prediction() -> None:
    with pytest.raises(ValueError):
        SimpleCNNBinaryClassifier().predict(np.zeros((1, 8, 8)))
