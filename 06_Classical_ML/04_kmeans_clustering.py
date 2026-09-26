import numpy as np


class KMeansScratch:
    def __init__(self, k: int = 3, max_iters: int = 100, tolerance: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.inertia = 0.0

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape

        # Randomly select k unique points as initial centroids
        np.random.seed(42)
        initial_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[initial_indices].astype(float)

        for _ in range(self.max_iters):
            # Compute Euclidean distances from all samples to all centroids: shape (n_samples, k)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

            # Assign each sample to the closest centroid
            labels = np.argmin(distances, axis=1)

            # Recompute centroids as the mean of assigned points
            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx in range(self.k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) > 0:
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)
                else:
                    new_centroids[cluster_idx] = self.centroids[cluster_idx]

            # Convergence check: centroid shift below tolerance
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tolerance:
                break

        # Calculate final inertia: sum of squared distances to closest centroid
        final_distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        min_sq_distances = np.min(final_distances, axis=1) ** 2
        self.inertia = float(np.sum(min_sq_distances))

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic Dataset: 3 distinct 2D clusters
    cluster_1 = np.random.randn(40, 2) + np.array([-4.0, -4.0])
    cluster_2 = np.random.randn(40, 2) + np.array([0.0, 4.0])
    cluster_3 = np.random.randn(40, 2) + np.array([4.0, -4.0])
    X = np.vstack((cluster_1, cluster_2, cluster_3))

    kmeans = KMeansScratch(k=3, max_iters=100)
    kmeans.fit(X)
    labels = kmeans.predict(X)

    print("--- K-Means Clustering (From Scratch) ---")
    print(f"Dataset Shape: {X.shape}")
    print(f"Optimal Centroids Found:\n{kmeans.centroids}")
    print(f"Final Inertia (WCSS): {kmeans.inertia:.4f}")

    # Cluster point distribution check
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"Cluster {cluster_id}: {count} points assigned")