from __future__ import annotations

import numpy as np

from ml_from_scratch.neural_networks import SimpleCNNBinaryClassifier
from ml_from_scratch.utils import accuracy_score


def make_bar_images(samples_per_class: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(71)
    images = []
    labels = []

    for _ in range(samples_per_class):
        vertical = rng.normal(scale=0.08, size=(8, 8))
        vertical[:, 3:5] += 1.0
        images.append(vertical)
        labels.append(1)

        horizontal = rng.normal(scale=0.08, size=(8, 8))
        horizontal[3:5, :] += 1.0
        images.append(horizontal)
        labels.append(0)

    return np.array(images), np.array(labels)


def run_cnn_demo() -> None:
    X, y = make_bar_images()
    model = SimpleCNNBinaryClassifier(
        n_filters=6,
        kernel_size=3,
        pool_size=2,
        hidden_layer_sizes=(10,),
        learning_rate=0.15,
        n_iterations=1200,
        random_state=72,
    )
    model.fit(X, y)
    predictions = model.predict(X)

    print("Simple CNN Binary Classifier")
    print(f"  accuracy: {accuracy_score(y, predictions):.3f}")
    print(f"  feature dimension: {model._extract_features(X[:1]).shape[1]}")
    print(f"  sample probabilities: {model.predict_proba(X[:4]).round(3).tolist()}")


if __name__ == "__main__":
    run_cnn_demo()
