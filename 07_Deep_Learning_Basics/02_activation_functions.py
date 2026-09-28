import numpy as np


class Activations:
    """Implementations of standard non-linear activation functions and analytical gradients."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid formulation
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    @staticmethod
    def sigmoid_grad(x: np.ndarray) -> np.ndarray:
        s = Activations.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_grad(x: np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1.0 - t**2

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        # Shift trick to avoid numerical overflow
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    test_input = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])

    print("--- Activation Functions & Derivatives Verification ---")
    print("Input:", test_input)
    print("Sigmoid:\n ", Activations.sigmoid(test_input))
    print("Sigmoid Grad:\n ", Activations.sigmoid_grad(test_input))
    print("ReLU:\n ", Activations.relu(test_input))
    print("ReLU Grad:\n ", Activations.relu_grad(test_input))
    print("Softmax (Vector):\n ", Activations.softmax(test_input))