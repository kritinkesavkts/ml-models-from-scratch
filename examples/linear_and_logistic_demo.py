from __future__ import annotations

import numpy as np

from ml_from_scratch.linear_models import LinearRegressionGD, LogisticRegressionGD
from ml_from_scratch.utils import accuracy_score, mean_squared_error


def run_linear_regression_demo() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 2))
    y = 4.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5

    model = LinearRegressionGD(learning_rate=0.05, n_iterations=700)
    model.fit(X, y)
    predictions = model.predict(X)

    print("Linear Regression")
    print(f"  weights: {model.weights_.round(3)}")
    print(f"  bias: {model.bias_:.3f}")
    print(f"  mse: {mean_squared_error(y, predictions):.6f}")


def run_logistic_regression_demo() -> None:
    rng = np.random.default_rng(13)
    class_0 = rng.normal(loc=(-2.0, -2.0), scale=0.7, size=(80, 2))
    class_1 = rng.normal(loc=(2.0, 2.0), scale=0.7, size=(80, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * len(class_0) + [1] * len(class_1))

    model = LogisticRegressionGD(learning_rate=0.2, n_iterations=800)
    model.fit(X, y)
    predictions = model.predict(X)

    print("Logistic Regression")
    print(f"  weights: {model.weights_.round(3)}")
    print(f"  bias: {model.bias_:.3f}")
    print(f"  accuracy: {accuracy_score(y, predictions):.3f}")


if __name__ == "__main__":
    run_linear_regression_demo()
    run_logistic_regression_demo()
