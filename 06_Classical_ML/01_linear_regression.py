import numpy as np


class LinearRegressionScratch:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits model parameters using Batch Gradient Descent."""
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            # Forward pass: y_hat = Xw + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # Mean Squared Error Loss
            loss = np.mean((y_predicted - y) ** 2)
            self.loss_history.append(loss)

            # Gradient computation
            # dL/dw = (2/N) * X^T (y_hat - y)
            # dL/db = (2/N) * sum(y_hat - y)
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """Analytical closed-form solution: theta = (X^T X)^(-1) X^T y"""
        n_samples = X.shape[0]
        # Add bias column of ones
        X_b = np.c_[np.ones((n_samples, 1)), X]
        # Normal equation
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


# Evaluation Metrics
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - (ss_residual / ss_total))


if __name__ == "__main__":
    # Synthetic Linear Dataset: y = 3.5 * X + 12.0 + noise
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = (3.5 * X.squeeze() + 12.0 + np.random.randn(100) * 0.5)

    # 1. Train with Gradient Descent
    model_gd = LinearRegressionScratch(learning_rate=0.05, n_iterations=1500)
    model_gd.fit(X, y)
    y_pred_gd = model_gd.predict(X)

    # 2. Train with Closed-Form Normal Equation
    model_ne = LinearRegressionScratch()
    model_ne.fit_normal_equation(X, y)
    y_pred_ne = model_ne.predict(X)

    print("--- Linear Regression (From Scratch) ---")
    print("True Parameters: Weight = 3.5000, Bias = 12.0000")
    print(f"\n[Gradient Descent]")
    print(f"Learned Weight: {model_gd.weights[0]:.4f}")
    print(f"Learned Bias:   {model_gd.bias:.4f}")
    print(f"MSE:            {mean_squared_error(y, y_pred_gd):.4f}")
    print(f"R2 Score:       {r2_score(y, y_pred_gd):.4f}")

    print(f"\n[Normal Equation (Closed-form)]")
    print(f"Solved Weight:  {model_ne.weights[0]:.4f}")
    print(f"Solved Bias:    {model_ne.bias:.4f}")
    print(f"MSE:            {mean_squared_error(y, y_pred_ne):.4f}")
    print(f"R2 Score:       {r2_score(y, y_pred_ne):.4f}")