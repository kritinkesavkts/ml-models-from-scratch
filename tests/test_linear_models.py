from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.linear_models import LinearRegressionGD, LogisticRegressionGD
from ml_from_scratch.utils import accuracy_score, mean_squared_error


def test_linear_regression_learns_known_relationship() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 2))
    y = 3.0 * X[:, 0] - 5.0 * X[:, 1] + 2.0

    model = LinearRegressionGD(learning_rate=0.05, n_iterations=800)
    model.fit(X, y)

    predictions = model.predict(X)

    assert mean_squared_error(y, predictions) < 1e-6
    assert np.allclose(model.weights_, np.array([3.0, -5.0]), atol=1e-3)
    assert model.bias_ == pytest.approx(2.0, abs=1e-3)
    assert model.losses_[0] > model.losses_[-1]


def test_logistic_regression_separates_two_clusters() -> None:
    rng = np.random.default_rng(123)
    class_0 = rng.normal(loc=(-2.0, -2.0), scale=0.6, size=(100, 2))
    class_1 = rng.normal(loc=(2.0, 2.0), scale=0.6, size=(100, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * len(class_0) + [1] * len(class_1))

    model = LogisticRegressionGD(learning_rate=0.2, n_iterations=700)
    model.fit(X, y)

    predictions = model.predict(X)

    assert accuracy_score(y, predictions) >= 0.98
    assert model.losses_[0] > model.losses_[-1]


def test_models_require_fit_before_prediction() -> None:
    with pytest.raises(ValueError):
        LinearRegressionGD().predict(np.array([[1.0]]))

    with pytest.raises(ValueError):
        LogisticRegressionGD().predict(np.array([[1.0]]))
