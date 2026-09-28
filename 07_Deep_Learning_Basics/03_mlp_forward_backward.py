import numpy as np


class TwoLayerMLP:
    """Multi-Layer Perceptron (Input -> Hidden -> Output) using vectorized NumPy backpropagation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 0.1):
        self.lr = lr

        # He initialization for ReLU activation
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        # Xavier initialization for output layer
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0.0, self.Z1)  # ReLU
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        loss = float(np.mean((y_pred - y_true) ** 2))

        # Gradients at output layer (MSE)
        dZ2 = (2.0 / m) * (y_pred - y_true)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Gradients at hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = np.dot(self.X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # SGD parameter updates
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss


if __name__ == "__main__":
    np.random.seed(42)

    # 4 synthetic samples with 3 features each
    X = np.random.randn(4, 3)
    y = np.array([[1.0], [0.0], [0.0], [1.0]])

    model = TwoLayerMLP(input_dim=3, hidden_dim=4, output_dim=1, lr=0.05)

    print("--- Training 2-Layer MLP (Vectorized Backprop) ---")
    for epoch in range(1, 301):
        preds = model.forward(X)
        current_loss = model.backward(y, preds)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/300 - Loss: {current_loss:.6f}")

    print("\nFinal Predictions:\n", model.forward(X))
    print("Targets:\n", y)