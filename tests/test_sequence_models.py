from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.sequence_models import GRUCell, LSTMCell, RNNCell, SimpleRNNBinaryClassifier
from ml_from_scratch.utils import accuracy_score


def _make_sequence_data(n_samples: int = 220, sequence_length: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(401)
    X = rng.normal(size=(n_samples, sequence_length, 1))
    y = (X[:, -2:, 0].sum(axis=1) > X[:, :2, 0].sum(axis=1)).astype(int)
    return X, y


def test_rnn_cell_returns_expected_hidden_shape() -> None:
    cell = RNNCell(input_size=3, hidden_size=5, random_state=1)
    x_t = np.ones((4, 3))
    h_prev = np.zeros((4, 5))

    assert cell.forward(x_t, h_prev).shape == (4, 5)


def test_lstm_cell_returns_hidden_and_cell_states() -> None:
    cell = LSTMCell(input_size=3, hidden_size=5, random_state=2)
    x_t = np.ones((4, 3))
    h_prev = np.zeros((4, 5))
    c_prev = np.zeros((4, 5))

    h_t, c_t = cell.forward(x_t, h_prev, c_prev)

    assert h_t.shape == (4, 5)
    assert c_t.shape == (4, 5)


def test_gru_cell_returns_expected_hidden_shape() -> None:
    cell = GRUCell(input_size=3, hidden_size=5, random_state=3)
    x_t = np.ones((4, 3))
    h_prev = np.zeros((4, 5))

    assert cell.forward(x_t, h_prev).shape == (4, 5)


def test_simple_rnn_binary_classifier_learns_sequence_rule() -> None:
    X, y = _make_sequence_data()
    model = SimpleRNNBinaryClassifier(
        hidden_size=10,
        learning_rate=0.15,
        n_iterations=900,
        random_state=402,
    )

    model.fit(X, y)

    assert accuracy_score(y, model.predict(X)) >= 0.88
    assert model.losses_[0] > model.losses_[-1]


def test_simple_rnn_requires_fit_before_prediction() -> None:
    with pytest.raises(ValueError):
        SimpleRNNBinaryClassifier().predict(np.zeros((2, 3, 1)))


def test_simple_rnn_rejects_non_sequence_input() -> None:
    with pytest.raises(ValueError):
        SimpleRNNBinaryClassifier().fit(np.zeros((2, 3)), np.array([0, 1]))
