from __future__ import annotations

import numpy as np
import pytest

from ml_from_scratch.bayes import GaussianNaiveBayes
from ml_from_scratch.cluster import KMeans
from ml_from_scratch.decomposition import PCA
from ml_from_scratch.neighbors import KNNClassifier, KNNRegressor
from ml_from_scratch.utils import accuracy_score, mean_squared_error


def test_knn_classifier_predicts_by_neighbor_majority() -> None:
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = KNNClassifier(n_neighbors=3).fit(X, y)

    assert model.predict(np.array([[1.5], [9.5]])).tolist() == [0, 1]


def test_knn_regressor_averages_nearest_targets() -> None:
    X = np.array([[0.0], [2.0], [4.0]])
    y = np.array([0.0, 2.0, 4.0])

    model = KNNRegressor(n_neighbors=2).fit(X, y)

    assert mean_squared_error(np.array([1.0]), model.predict(np.array([[1.0]]))) == pytest.approx(0.0)


def test_gaussian_naive_bayes_separates_gaussian_clusters() -> None:
    rng = np.random.default_rng(11)
    class_0 = rng.normal(loc=(-2.0, -2.0), scale=0.5, size=(80, 2))
    class_1 = rng.normal(loc=(2.0, 2.0), scale=0.5, size=(80, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * len(class_0) + [1] * len(class_1))

    model = GaussianNaiveBayes().fit(X, y)

    assert accuracy_score(y, model.predict(X)) >= 0.98


def test_kmeans_finds_two_cluster_centers() -> None:
    rng = np.random.default_rng(17)
    cluster_a = rng.normal(loc=(-4.0, 0.0), scale=0.3, size=(50, 2))
    cluster_b = rng.normal(loc=(4.0, 0.0), scale=0.3, size=(50, 2))
    X = np.vstack([cluster_a, cluster_b])

    model = KMeans(n_clusters=2, random_state=9).fit(X)
    centers = model.centroids_[np.argsort(model.centroids_[:, 0])]

    assert centers[0, 0] == pytest.approx(-4.0, abs=0.25)
    assert centers[1, 0] == pytest.approx(4.0, abs=0.25)
    assert model.inertia_ < 30.0


def test_pca_reduces_correlated_features_to_main_direction() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=120)
    X = np.column_stack([x, 3.0 * x + rng.normal(scale=0.03, size=120)])

    model = PCA(n_components=1)
    X_reduced = model.fit_transform(X)
    X_reconstructed = model.inverse_transform(X_reduced)

    assert X_reduced.shape == (120, 1)
    assert model.explained_variance_ratio_[0] > 0.99
    assert mean_squared_error(X, X_reconstructed) < 0.001


def test_classical_models_require_fit_before_prediction() -> None:
    with pytest.raises(ValueError):
        KNNClassifier().predict(np.array([[1.0]]))
    with pytest.raises(ValueError):
        GaussianNaiveBayes().predict(np.array([[1.0]]))
    with pytest.raises(ValueError):
        KMeans().predict(np.array([[1.0]]))
    with pytest.raises(ValueError):
        PCA(n_components=1).transform(np.array([[1.0]]))
