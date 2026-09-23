import math
import time
import numpy as np


# --- Pure Python Implementations (From Scratch) ---

def dot_product_python(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    total = 0.0
    for a, b in zip(v1, v2):
        total += a * b
    return total


def l2_norm_python(v: list[float]) -> float:
    squared_sum = 0.0
    for val in v:
        squared_sum += val * val
    return math.sqrt(squared_sum)


def cosine_similarity_python(v1: list[float], v2: list[float]) -> float:
    norm1 = l2_norm_python(v1)
    norm2 = l2_norm_python(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product_python(v1, v2) / (norm1 * norm2)


# --- NumPy Vectorized Implementations ---

def vector_ops_numpy(v1_np: np.ndarray, v2_np: np.ndarray):
    dot = np.dot(v1_np, v2_np)
    norm1 = np.linalg.norm(v1_np)
    norm2 = np.linalg.norm(v2_np)
    cos_sim = dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
    return dot, norm1, cos_sim


if __name__ == "__main__":
    # Sanity check with small geometric vectors
    a = [3.0, 4.0]
    b = [4.0, 3.0]

    print("--- Basic Vector Operations ---")
    print("Vector A:", a)
    print("Vector B:", b)
    print(f"Dot Product:        {dot_product_python(a, b)}")
    print(f"L2 Norm of A:       {l2_norm_python(a)}")
    print(f"Cosine Similarity:  {cosine_similarity_python(a, b):.4f}")

    # Benchmark: 1,000,000 dimensional embedding vectors
    dim = 1_000_000
    vec_a = [1.0] * dim
    vec_b = [2.0] * dim

    vec_a_np = np.array(vec_a, dtype=np.float32)
    vec_b_np = np.array(vec_b, dtype=np.float32)

    # Pure Python benchmark
    t0 = time.time()
    dot_py = dot_product_python(vec_a, vec_b)
    t_python = time.time() - t0

    # NumPy benchmark
    t0 = time.time()
    dot_np = np.dot(vec_a_np, vec_b_np)
    t_numpy = time.time() - t0

    print(f"\n--- Benchmark ({dim:,} Dimensions) ---")
    print(f"Pure Python Loop: {t_python:.4f}s")
    print(f"NumPy Vectorized: {t_numpy:.6f}s")
    print(f"Speedup: ~{t_python / max(t_numpy, 1e-9):.1f}x faster")