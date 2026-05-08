from __future__ import annotations

import numpy as np

from ml_from_scratch.sequence_models import GRUCell, LSTMCell, RNNCell, SimpleRNNBinaryClassifier
from ml_from_scratch.utils import accuracy_score


def make_sequence_data(n_samples: int = 240, sequence_length: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(81)
    X = rng.normal(size=(n_samples, sequence_length, 1))
    y = (X[:, -2:, 0].sum(axis=1) > X[:, :2, 0].sum(axis=1)).astype(int)
    return X, y


def run_cell_demo() -> None:
    x_t = np.array([[0.2, -0.1]])
    h_prev = np.zeros((1, 4))
    c_prev = np.zeros((1, 4))

    rnn_h = RNNCell(input_size=2, hidden_size=4, random_state=82).forward(x_t, h_prev)
    lstm_h, lstm_c = LSTMCell(input_size=2, hidden_size=4, random_state=83).forward(
        x_t, h_prev, c_prev
    )
    gru_h = GRUCell(input_size=2, hidden_size=4, random_state=84).forward(x_t, h_prev)

    print("Recurrent Cells")
    print(f"  RNN hidden shape: {rnn_h.shape}")
    print(f"  LSTM hidden/cell shapes: {lstm_h.shape}, {lstm_c.shape}")
    print(f"  GRU hidden shape: {gru_h.shape}")


def run_rnn_classifier_demo() -> None:
    X, y = make_sequence_data()
    model = SimpleRNNBinaryClassifier(
        hidden_size=10,
        learning_rate=0.15,
        n_iterations=900,
        random_state=85,
    )
    model.fit(X, y)
    predictions = model.predict(X)

    print("Simple RNN Binary Classifier")
    print(f"  accuracy: {accuracy_score(y, predictions):.3f}")
    print(f"  loss improvement: {model.losses_[0]:.3f} -> {model.losses_[-1]:.3f}")
    print(f"  sample probabilities: {model.predict_proba(X[:5]).round(3).tolist()}")


if __name__ == "__main__":
    run_cell_demo()
    run_rnn_classifier_demo()
