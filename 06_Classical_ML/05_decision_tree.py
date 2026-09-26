from collections import Counter
from typing import Optional
import numpy as np


class Node:
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        *,
        value: Optional[int] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifierScratch:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini_impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return 1.0 - sum(p ** 2 for p in probabilities)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = -1.0
        best_feature, best_threshold = None, None
        current_impurity = self._gini_impurity(y)
        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_y, right_y = y[left_mask], y[right_mask]
                n_left, n_right = len(left_y), len(right_y)

                # Weighted Gini Impurity
                left_weight = n_left / n_samples
                right_weight = n_right / n_samples
                child_impurity = (left_weight * self._gini_impurity(left_y)) + (
                    right_weight * self._gini_impurity(right_y)
                )

                # Impurity decrease (information gain proxy)
                impurity_gain = current_impurity - child_impurity

                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria: pure node, max depth, or insufficient samples
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for x in X])


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic non-linear classification dataset
    cluster_a = np.random.randn(40, 2) + np.array([-1.5, -1.5])
    cluster_b = np.random.randn(40, 2) + np.array([1.5, 1.5])

    X = np.vstack((cluster_a, cluster_b))
    y = np.array([0] * 40 + [1] * 40)

    # Train/Test Split (80/20)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int(0.8 * len(y))

    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_test, y_test = X[indices[split:]], y[indices[split:]]

    tree = DecisionTreeClassifierScratch(max_depth=3, min_samples_split=2)
    tree.fit(X_train, y_train)

    predictions = tree.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print("--- Decision Tree Classifier (CART Scratch) ---")
    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"Root Split Feature Index: {tree.root.feature}")
    print(f"Root Split Threshold:     {tree.root.threshold:.4f}")
    print(f"Test Accuracy:            {acc * 100:.2f}%")