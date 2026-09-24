import math
from typing import List, Tuple
import numpy as np


# --- Pure Python Implementations (From Scratch) ---

def calculate_mean(values: List[float]) -> float:
    if not values:
        raise ValueError("List cannot be empty")
    return sum(values) / len(values)


def calculate_variance(values: List[float], sample: bool = True) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("Variance requires at least two data points")
    mean = calculate_mean(values)
    squared_diff_sum = sum((x - mean) ** 2 for x in values)
    # Bessel's correction: divide by (n - 1) for sample variance
    divisor = (n - 1) if sample else n
    return squared_diff_sum / divisor


def calculate_std(values: List[float], sample: bool = True) -> float:
    return math.sqrt(calculate_variance(values, sample=sample))


def calculate_covariance(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Lists must be of equal length")
    n = len(x)
    if n < 2:
        raise ValueError("Covariance requires at least two data points")

    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)
    cov_sum = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    return cov_sum / (n - 1)


def gaussian_pdf(x: float, mean: float, std: float) -> float:
    """Calculates Probability Density Function of Normal Distribution."""
    if std <= 0:
        raise ValueError("Standard deviation must be strictly positive")
    exponent = math.exp(-0.5 * ((x - mean) / std) ** 2)
    return (1.0 / (std * math.sqrt(2.0 * math.pi))) * exponent


if __name__ == "__main__":
    # Test Data: Feature X and Feature Y
    data_x = [2.0, 4.0, 6.0, 8.0, 10.0]
    data_y = [1.0, 3.0, 5.0, 7.0, 9.0]

    # Scratch Calculations
    mean_x = calculate_mean(data_x)
    var_x = calculate_variance(data_x, sample=True)
    std_x = calculate_std(data_x, sample=True)
    cov_xy = calculate_covariance(data_x, data_y)

    # NumPy Verifications
    np_mean_x = np.mean(data_x)
    np_var_x = np.var(data_x, ddof=1)
    np_cov_matrix = np.cov(data_x, data_y)

    print("--- Statistical Foundations: Pure Python vs NumPy ---")
    print(f"Mean (Scratch):       {mean_x:.4f} | NumPy: {np_mean_x:.4f}")
    print(f"Sample Var (Scratch): {var_x:.4f} | NumPy: {np_var_x:.4f}")
    print(f"Sample Std (Scratch): {std_x:.4f} | NumPy: {np.std(data_x, ddof=1):.4f}")
    print(f"Covariance(X, Y):     {cov_xy:.4f} | NumPy: {np_cov_matrix[0, 1]:.4f}")

    # Gaussian PDF Evaluation
    test_val = 6.0
    prob_density = gaussian_pdf(test_val, mean=mean_x, std=std_x)
    print("\n--- Gaussian Normal Distribution PDF ---")
    print(f"P(X = {test_val} | mu={mean_x:.1f}, sigma={std_x:.2f}) = {prob_density:.5f}")