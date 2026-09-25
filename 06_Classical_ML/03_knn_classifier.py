import numpy as np
from collections import Counter


class KNNClassifierScratch:
    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Memorizes training instances (lazy learning)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))

    def _predict_single(self, x: np.ndarray) -> int:
        # Compute distances from x to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]

        # Extract labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return majority vote label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    np.random.seed(42)

    # 3-Class Synthetic Dataset (2D points for 3 distinct clusters)
    cluster_0 = np.random.randn(30, 2) + np.array([-3.0, -3.0])
    cluster_1 = np.random.randn(30, 2) + np.array([3.0, 3.0])
    cluster_2 = np.random.randn(30, 2) + np.array([3.0, -3.0])

    X = np.vstack((cluster_0, cluster_1, cluster_2))
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)

    # Train-test split (80% train, 20% test)
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(y))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Evaluate KNN for different values of k
    print("--- K-Nearest Neighbors Classifier (From Scratch) ---")
    print(f"Total Samples: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}\n")

    for k_val in [1, 3, 5, 7]:
        knn = KNNClassifierScratch(k=k_val)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"KNN (k={k_val}) -> Test Accuracy: {acc * 100:.2f}%")