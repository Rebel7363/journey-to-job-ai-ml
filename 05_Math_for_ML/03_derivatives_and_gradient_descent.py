import math
from typing import Callable, Tuple, List


# 1. Finite Difference Numerical Derivative
def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Computes the central difference approximation of f'(x)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


# 2. 1D Gradient Descent Optimizer
def gradient_descent_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    initial_x: float,
    learning_rate: float = 0.1,
    max_iters: int = 50,
    tolerance: float = 1e-6
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """
    Minimizes 1D convex function f(x) using analytical gradient df(x).
    Returns optimal x and iteration history.
    """
    x = initial_x
    history = []

    for i in range(max_iters):
        grad = df(x)
        history.append((i, x, f(x)))

        if abs(grad) < tolerance:
            break

        # Parameter update rule: x_new = x_old - lr * gradient
        x = x - learning_rate * grad

    return x, history


# 3. 2D Multivariate Gradient Descent
def gradient_descent_2d(
    initial_point: Tuple[float, float],
    learning_rate: float = 0.1,
    max_iters: int = 50,
    tolerance: float = 1e-6
) -> Tuple[Tuple[float, float], float]:
    """
    Minimizes f(x, y) = x^2 + 2y^2
    Gradient: [df/dx, df/dy] = [2x, 4y]
    Global minimum is at (0, 0) with value 0.
    """
    x, y = initial_point

    for _ in range(max_iters):
        grad_x = 2.0 * x
        grad_y = 4.0 * y

        if math.sqrt(grad_x**2 + grad_y**2) < tolerance:
            break

        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

    min_value = x**2 + 2.0 * (y**2)
    return (x, y), min_value


if __name__ == "__main__":
    # Test 1: Analytical vs Numerical Derivative
    # f(x) = x^3 - 2x + 4 -> f'(x) = 3x^2 - 2
    f_test = lambda x: x**3 - 2.0 * x + 4.0
    df_analytical = lambda x: 3.0 * (x**2) - 2.0

    test_point = 3.0
    num_grad = numerical_derivative(f_test, test_point)
    ana_grad = df_analytical(test_point)

    print("--- Derivative Verification ---")
    print(f"Point x = {test_point}")
    print(f"Numerical Derivative (Central Difference): {num_grad:.6f}")
    print(f"Analytical Derivative (Exact):             {ana_grad:.6f}")
    print(f"Approximation Error:                       {abs(num_grad - ana_grad):.2e}")

    # Test 2: 1D Gradient Descent Optimization
    # Loss: L(w) = w^2 - 4w + 7, Minimum at w = 2, L(2) = 3
    loss_fn = lambda w: w**2 - 4.0 * w + 7.0
    d_loss_fn = lambda w: 2.0 * w - 4.0

    start_w = 10.0
    opt_w, history = gradient_descent_1d(loss_fn, d_loss_fn, initial_x=start_w, learning_rate=0.1)

    print("\n--- 1D Optimization Progress ---")
    print(f"Starting at w = {start_w:.2f}")
    for step, w_val, cost in history[:6]:
        print(f"Iteration {step:02d}: w = {w_val:.4f}, Cost = {cost:.4f}")
    print(f"Converged at: w = {opt_w:.4f} (Optimal Target: 2.0000)")

    # Test 3: 2D Multivariate Optimization
    start_2d = (5.0, -4.0)
    (opt_x, opt_y), min_val = gradient_descent_2d(start_2d, learning_rate=0.1)

    print("\n--- 2D Multivariate Optimization ---")
    print(f"Start Point: (x={start_2d[0]}, y={start_2d[1]})")
    print(f"Optimized Point: (x={opt_x:.6f}, y={opt_y:.6f})")
    print(f"Minimum Function Value: {min_val:.6e}")