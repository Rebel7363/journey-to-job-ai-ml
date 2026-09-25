import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid to prevent overflow
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Binary Cross-Entropy (Log Loss) with epsilon clipping for numerical stability
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_predicted, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)

            # Gradient computation: dL/dw = (1/N) * X^T (y_hat - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


# Classification Evaluation Metrics
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic binary classification data: 2 clusters in 2D space
    class_0 = np.random.randn(50, 2) + np.array([-2.0, -2.0])
    class_1 = np.random.randn(50, 2) + np.array([2.0, 2.0])

    X = np.vstack((class_0, class_1))
    y = np.array([0] * 50 + [1] * 50)

    # Shuffle dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Train model
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    tp, tn, fp, fn = confusion_matrix(y, y_pred)

    print("--- Logistic Regression (From Scratch) ---")
    print(f"Final Weights: {model.weights}")
    print(f"Final Bias:    {model.bias:.4f}")
    print(f"Initial Loss:  {model.loss_history[0]:.4f}")
    print(f"Final Loss:    {model.loss_history[-1]:.4f}")
    print(f"Accuracy:      {acc * 100:.2f}%")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")