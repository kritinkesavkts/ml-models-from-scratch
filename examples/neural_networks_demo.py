from __future__ import annotations

import numpy as np

from ml_from_scratch.neural_networks import MLPBinaryClassifier
from ml_from_scratch.utils import accuracy_score


def make_xor_data(repeats: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(61)
    base_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    base_y = np.array([0, 1, 1, 0])
    X = np.repeat(base_X, repeats, axis=0)
    y = np.repeat(base_y, repeats)
    X = X + rng.normal(scale=0.08, size=X.shape)
    return X, y


def run_mlp_demo() -> None:
    X, y = make_xor_data()
    model = MLPBinaryClassifier(
        hidden_layer_sizes=(8, 4),
        learning_rate=0.2,
        n_iterations=2500,
        random_state=62,
    )
    model.fit(X, y)
    predictions = model.predict(X)

    print("MLP Binary Classifier")
    print(f"  accuracy: {accuracy_score(y, predictions):.3f}")
    print(f"  loss improvement: {model.losses_[0]:.3f} -> {model.losses_[-1]:.3f}")
    print(f"  sample probabilities: {model.predict_proba(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).round(3).tolist()}")


if __name__ == "__main__":
    run_mlp_demo()
