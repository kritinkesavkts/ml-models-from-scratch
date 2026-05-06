from __future__ import annotations

import numpy as np

from ml_from_scratch.bayes import GaussianNaiveBayes
from ml_from_scratch.cluster import KMeans
from ml_from_scratch.decomposition import PCA
from ml_from_scratch.neighbors import KNNClassifier, KNNRegressor
from ml_from_scratch.utils import accuracy_score, mean_squared_error


def run_knn_demo() -> None:
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y_cls = np.array([0, 0, 0, 1, 1, 1])
    y_reg = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])

    classifier = KNNClassifier(n_neighbors=3).fit(X, y_cls)
    regressor = KNNRegressor(n_neighbors=2).fit(X, y_reg)

    print("KNN")
    print(f"  classifier predictions: {classifier.predict(np.array([[1.5], [9.5]])).tolist()}")
    print(f"  regressor mse: {mean_squared_error(np.array([1.5, 9.5]), regressor.predict(np.array([[1.5], [9.5]]))):.3f}")


def run_naive_bayes_demo() -> None:
    rng = np.random.default_rng(21)
    class_0 = rng.normal(loc=(-1.5, -1.5), scale=0.5, size=(60, 2))
    class_1 = rng.normal(loc=(1.5, 1.5), scale=0.5, size=(60, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * len(class_0) + [1] * len(class_1))

    model = GaussianNaiveBayes().fit(X, y)
    predictions = model.predict(X)

    print("Gaussian Naive Bayes")
    print(f"  accuracy: {accuracy_score(y, predictions):.3f}")


def run_kmeans_demo() -> None:
    rng = np.random.default_rng(31)
    cluster_a = rng.normal(loc=(-3.0, 0.0), scale=0.4, size=(40, 2))
    cluster_b = rng.normal(loc=(3.0, 0.0), scale=0.4, size=(40, 2))
    X = np.vstack([cluster_a, cluster_b])

    model = KMeans(n_clusters=2, random_state=5).fit(X)

    print("K-Means")
    print(f"  centroids: {model.centroids_.round(2).tolist()}")
    print(f"  inertia: {model.inertia_:.3f}")


def run_pca_demo() -> None:
    rng = np.random.default_rng(41)
    x = rng.normal(size=100)
    X = np.column_stack([x, 2.0 * x + rng.normal(scale=0.05, size=100)])

    pca = PCA(n_components=1)
    X_reduced = pca.fit_transform(X)

    print("PCA")
    print(f"  reduced shape: {X_reduced.shape}")
    print(f"  explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")


if __name__ == "__main__":
    run_knn_demo()
    run_naive_bayes_demo()
    run_kmeans_demo()
    run_pca_demo()
