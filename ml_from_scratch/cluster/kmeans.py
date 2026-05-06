from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KMeans:
    """K-Means clustering with random centroid initialization."""

    n_clusters: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    random_state: int | None = None
    centroids_: np.ndarray | None = field(default=None, init=False)
    labels_: np.ndarray | None = field(default=None, init=False)
    inertia_: float | None = field(default=None, init=False)
    n_iter_: int = field(default=0, init=False)

    def fit(self, X: np.ndarray) -> "KMeans":
        X = self._validate_X(X)
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed number of samples.")

        rng = np.random.default_rng(self.random_state)
        initial_idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.centroids_ = X[initial_idx].copy()

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_labels(X, self.centroids_)
            new_centroids = self._update_centroids(X, labels, rng)
            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.labels_ = labels
            self.n_iter_ = iteration
            if shift <= self.tol:
                break

        self.labels_ = self._assign_labels(X, self.centroids_)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X = self._validate_X(X)
        return self._assign_labels(X, self.centroids_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        for cluster_id in range(self.n_clusters):
            members = X[labels == cluster_id]
            if len(members) == 0:
                centroids[cluster_id] = X[rng.integers(0, X.shape[0])]
            else:
                centroids[cluster_id] = members.mean(axis=0)
        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        distances = X - self.centroids_[labels]
        return float(np.sum(distances**2))

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        return X
