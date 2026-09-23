# Module 05: Mathematical Foundations for Machine Learning

This module focuses on the mathematical bedrock of AI/ML systems. It covers foundational Linear Algebra operations implemented from scratch in pure Python alongside vectorized, cache-optimized NumPy comparisons.

## Implementations & Benchmarks

| Topic | Script | Scratch Complexity | Hardware / Vectorized Implementation | Key Machine Learning Application |
|:---|:---|:---:|:---:|:---|
| **Vector Operations** | [`01_vector_operations.py`](./01_vector_operations.py) | $O(n)$ | SIMD / NumPy Vectorization | Embeddings, Cosine Similarity, Loss Functions |
| **Matrix Multiplication** | [`02_matrix_multiplication.py`](./02_matrix_multiplication.py) | $O(n^3)$ | BLAS / GEMM C-Extensions | Neural Network Dense Layers, Feedforward Passes |

## Theoretical Invariants

1. **Dot Product & Cosine Similarity:**
   * Dot product computes directional projection: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i$.
   * Cosine similarity normalizes magnitude using L2 norm to measure angular orientation:
     $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2}$$
   * Core building block for text embedding search, semantic ranking, and recommendation systems.

2. **Matrix Multiplication ($C = A \times B$):**
   * Inner dimensions must match: $(m \times k) \times (k \times n) \to (m \times n)$.
   * Triple-loop scratch implementations incur $O(n^3)$ computational complexity and suffer from cache misses.
   * High-performance libraries use BLAS/GEMM routines, memory-tiling, and SIMD parallelization to achieve massive throughput gains.