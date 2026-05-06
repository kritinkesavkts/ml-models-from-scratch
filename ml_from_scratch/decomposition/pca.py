from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PCA:
    """Principal Component Analysis using covariance eigendecomposition."""

    n_components: int
    components_: np.ndarray | None = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    explained_variance_: np.ndarray | None = field(default=None, init=False)
    explained_variance_ratio_: np.ndarray | None = field(default=None, init=False)

    def fit(self, X: np.ndarray) -> "PCA":
        X = self._validate_X(X)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed number of features.")

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        covariance = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Model must be fitted before transform.")
        X = self._validate_X(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Model must be fitted before inverse_transform.")
        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(1, -1)
        return X_transformed @ self.components_ + self.mean_

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        return X
