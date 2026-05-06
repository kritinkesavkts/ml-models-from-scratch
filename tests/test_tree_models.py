from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.tree_models import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from ml_from_scratch.utils import accuracy_score, mean_squared_error


def _make_classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(101)
    X = rng.normal(size=(180, 2))
    y = ((X[:, 0] > 0.2) | (X[:, 1] < -0.8)).astype(int)
    return X, y


def _make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(102)
    X = rng.uniform(-2.0, 2.0, size=(180, 1))
    y = np.sin(3.0 * X[:, 0]) + 0.2 * X[:, 0]
    return X, y


def test_decision_tree_classifier_learns_axis_aligned_rule() -> None:
    X, y = _make_classification_data()

    model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X, y)

    assert accuracy_score(y, model.predict(X)) >= 0.95


def test_decision_tree_regressor_learns_piecewise_regression() -> None:
    X, y = _make_regression_data()

    model = DecisionTreeRegressor(max_depth=5, random_state=2).fit(X, y)

    assert mean_squared_error(y, model.predict(X)) < 0.04


def test_random_forest_classifier_combines_multiple_trees() -> None:
    X, y = _make_classification_data()

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=5,
        random_state=3,
    ).fit(X, y)

    assert len(model.trees_) == 20
    assert accuracy_score(y, model.predict(X)) >= 0.93


def test_gradient_boosting_regressor_reduces_residual_error() -> None:
    X, y = _make_regression_data()

    model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.12,
        max_depth=2,
        random_state=4,
    ).fit(X, y)

    baseline_mse = mean_squared_error(y, np.full_like(y, np.mean(y)))

    assert model.losses_[0] < baseline_mse
    assert model.losses_[-1] < 0.04
    assert model.losses_[0] > model.losses_[-1]


def test_tree_models_require_fit_before_prediction() -> None:
    X = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError):
        DecisionTreeClassifier().predict(X)
    with pytest.raises(ValueError):
        DecisionTreeRegressor().predict(X)
    with pytest.raises(ValueError):
        RandomForestClassifier().predict(X)
    with pytest.raises(ValueError):
        GradientBoostingRegressor().predict(X)
