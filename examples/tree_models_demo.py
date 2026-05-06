from __future__ import annotations

import numpy as np

from ml_from_scratch.tree_models import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from ml_from_scratch.utils import accuracy_score, mean_squared_error


def make_classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(52)
    X = rng.normal(size=(160, 2))
    y = ((X[:, 0] > 0.0) & (X[:, 1] > -0.4)).astype(int)
    return X, y


def make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(53)
    X = rng.uniform(-2.0, 2.0, size=(180, 1))
    y = np.sin(2.5 * X[:, 0]) + 0.25 * X[:, 0]
    return X, y


def run_decision_tree_demo() -> None:
    X_cls, y_cls = make_classification_data()
    classifier = DecisionTreeClassifier(max_depth=3).fit(X_cls, y_cls)

    X_reg, y_reg = make_regression_data()
    regressor = DecisionTreeRegressor(max_depth=4).fit(X_reg, y_reg)

    print("Decision Tree")
    print(f"  classifier accuracy: {accuracy_score(y_cls, classifier.predict(X_cls)):.3f}")
    print(f"  regressor mse: {mean_squared_error(y_reg, regressor.predict(X_reg)):.3f}")


def run_random_forest_demo() -> None:
    X, y = make_classification_data()
    model = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=54)
    model.fit(X, y)

    print("Random Forest")
    print(f"  accuracy: {accuracy_score(y, model.predict(X)):.3f}")


def run_gradient_boosting_demo() -> None:
    X, y = make_regression_data()
    model = GradientBoostingRegressor(
        n_estimators=40,
        learning_rate=0.15,
        max_depth=2,
        random_state=55,
    )
    model.fit(X, y)

    print("Gradient Boosting")
    print(f"  final training mse: {mean_squared_error(y, model.predict(X)):.3f}")
    print(f"  loss improvement: {model.losses_[0]:.3f} -> {model.losses_[-1]:.3f}")


if __name__ == "__main__":
    run_decision_tree_demo()
    run_random_forest_demo()
    run_gradient_boosting_demo()
