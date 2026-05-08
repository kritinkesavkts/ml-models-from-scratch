from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.neural_networks.activations import sigmoid


@dataclass
class RNNCell:
    """Vanilla recurrent cell.

    h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)
    """

    input_size: int
    hidden_size: int
    random_state: int | None = None
    W_xh: np.ndarray = field(init=False)
    W_hh: np.ndarray = field(init=False)
    b_h: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / max(1, self.input_size))
        self.W_xh = rng.normal(0.0, scale, size=(self.input_size, self.hidden_size))
        self.W_hh = rng.normal(0.0, scale, size=(self.hidden_size, self.hidden_size))
        self.b_h = np.zeros((1, self.hidden_size))

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        x_t = _validate_step(x_t, self.input_size)
        h_prev = _validate_hidden(h_prev, self.hidden_size)
        return np.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)


@dataclass
class LSTMCell:
    """Long short-term memory cell with input, forget, output, and candidate gates."""

    input_size: int
    hidden_size: int
    random_state: int | None = None
    W_i: np.ndarray = field(init=False)
    W_f: np.ndarray = field(init=False)
    W_o: np.ndarray = field(init=False)
    W_g: np.ndarray = field(init=False)
    b_i: np.ndarray = field(init=False)
    b_f: np.ndarray = field(init=False)
    b_o: np.ndarray = field(init=False)
    b_g: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_state)
        concat_size = self.input_size + self.hidden_size
        scale = np.sqrt(1.0 / max(1, concat_size))
        self.W_i = rng.normal(0.0, scale, size=(concat_size, self.hidden_size))
        self.W_f = rng.normal(0.0, scale, size=(concat_size, self.hidden_size))
        self.W_o = rng.normal(0.0, scale, size=(concat_size, self.hidden_size))
        self.W_g = rng.normal(0.0, scale, size=(concat_size, self.hidden_size))
        self.b_i = np.zeros((1, self.hidden_size))
        self.b_f = np.ones((1, self.hidden_size))
        self.b_o = np.zeros((1, self.hidden_size))
        self.b_g = np.zeros((1, self.hidden_size))

    def forward(
        self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_t = _validate_step(x_t, self.input_size)
        h_prev = _validate_hidden(h_prev, self.hidden_size)
        c_prev = _validate_hidden(c_prev, self.hidden_size)
        combined = np.concatenate([x_t, h_prev], axis=1)

        input_gate = sigmoid(combined @ self.W_i + self.b_i)
        forget_gate = sigmoid(combined @ self.W_f + self.b_f)
        output_gate = sigmoid(combined @ self.W_o + self.b_o)
        candidate = np.tanh(combined @ self.W_g + self.b_g)

        c_t = forget_gate * c_prev + input_gate * candidate
        h_t = output_gate * np.tanh(c_t)
        return h_t, c_t


@dataclass
class GRUCell:
    """Gated recurrent unit cell with update and reset gates."""

    input_size: int
    hidden_size: int
    random_state: int | None = None
    W_z: np.ndarray = field(init=False)
    W_r: np.ndarray = field(init=False)
    W_h: np.ndarray = field(init=False)
    b_z: np.ndarray = field(init=False)
    b_r: np.ndarray = field(init=False)
    b_h: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_state)
        concat_size = self.input_size + self.hidden_size
        scale = np.sqrt(1.0 / max(1, concat_size))
        self.W_z = rng.normal(0.0, scale, size=(concat_size, self.hidden_size))
        self.W_r = rng.normal(0.0, scale, size=(concat_size, self.hidden_size))
        self.W_h = rng.normal(0.0, scale, size=(concat_size, self.hidden_size))
        self.b_z = np.zeros((1, self.hidden_size))
        self.b_r = np.zeros((1, self.hidden_size))
        self.b_h = np.zeros((1, self.hidden_size))

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        x_t = _validate_step(x_t, self.input_size)
        h_prev = _validate_hidden(h_prev, self.hidden_size)
        combined = np.concatenate([x_t, h_prev], axis=1)
        update_gate = sigmoid(combined @ self.W_z + self.b_z)
        reset_gate = sigmoid(combined @ self.W_r + self.b_r)
        candidate_input = np.concatenate([x_t, reset_gate * h_prev], axis=1)
        candidate = np.tanh(candidate_input @ self.W_h + self.b_h)
        return (1.0 - update_gate) * h_prev + update_gate * candidate


def _validate_step(x_t: np.ndarray, input_size: int) -> np.ndarray:
    x_t = np.asarray(x_t, dtype=float)
    if x_t.ndim == 1:
        x_t = x_t.reshape(1, -1)
    if x_t.ndim != 2 or x_t.shape[1] != input_size:
        raise ValueError(f"x_t must have shape (batch, {input_size}).")
    return x_t


def _validate_hidden(h_t: np.ndarray, hidden_size: int) -> np.ndarray:
    h_t = np.asarray(h_t, dtype=float)
    if h_t.ndim == 1:
        h_t = h_t.reshape(1, -1)
    if h_t.ndim != 2 or h_t.shape[1] != hidden_size:
        raise ValueError(f"hidden state must have shape (batch, {hidden_size}).")
    return h_t
