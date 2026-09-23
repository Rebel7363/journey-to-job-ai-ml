import time
import numpy as np


# Pure Python Implementation: O(n^3) Triple Nested Loop
def matrix_multiply_python(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Cannot multiply: Incompatible inner matrix dimensions.")

    # Initialize result matrix with zeros
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Triple loop: O(rows_A * cols_B * cols_A)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


if __name__ == "__main__":
    # Small dimension verification: (2x3) * (3x2) -> (2x2)
    mat_a = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    mat_b = [
        [7.0, 8.0],
        [9.0, 1.0],
        [2.0, 3.0]
    ]

    print("--- Matrix Multiplication Sanity Check ---")
    res_py = matrix_multiply_python(mat_a, mat_b)
    print("Pure Python Result:")
    for row in res_py:
        print(row)

    # Benchmark: 200x200 Square Matrix Multiplication
    size = 200
    np_A = np.random.randn(size, size).astype(np.float32)
    np_B = np.random.randn(size, size).astype(np.float32)

    list_A = np_A.tolist()
    list_B = np_B.tolist()

    # Benchmark Pure Python
    t0 = time.time()
    _ = matrix_multiply_python(list_A, list_B)
    t_python = time.time() - t0

    # Benchmark NumPy (BLAS / C optimization)
    t0 = time.time()
    _ = np.dot(np_A, np_B)
    t_numpy = time.time() - t0

    print(f"\n--- Benchmark ({size}x{size} Matrices) ---")
    print(f"Pure Python Loop O(n^3): {t_python:.4f}s")
    print(f"NumPy Dot (BLAS/C):       {t_numpy:.6f}s")
    print(f"Speedup: ~{t_python / max(t_numpy, 1e-9):.1f}x faster")