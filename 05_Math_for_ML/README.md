# Module 05: Mathematical Foundations for Machine Learning

This module focuses on the mathematical bedrock of AI/ML systems. It covers foundational Linear Algebra, Multivariable Calculus, Gradient Optimization, and Statistical Estimators implemented from scratch in pure Python alongside vectorized, cache-optimized NumPy comparisons.

## Implementations & Benchmarks

| Topic | Script | Scratch Complexity | Hardware / Vectorized Implementation | Key Machine Learning Application |
|:---|:---|:---:|:---:|:---|
| **Vector Operations** | [`01_vector_operations.py`](./01_vector_operations.py) | $O(n)$ | SIMD / NumPy Vectorization | Embeddings, Cosine Similarity, Loss Functions |
| **Matrix Multiplication** | [`02_matrix_multiplication.py`](./02_matrix_multiplication.py) | $O(n^3)$ | BLAS / GEMM C-Extensions | Neural Network Dense Layers, Feedforward Passes |
| **Gradients & Optimization** | [`03_derivatives_and_gradient_descent.py`](./03_derivatives_and_gradient_descent.py) | $O(k)$ per step | Analytical Gradients | Backpropagation, SGD, Model Parameter Updates |
| **Probability & Statistics** | [`04_probability_and_statistics.py`](./04_probability_and_statistics.py) | $O(n)$ | NumPy Statistical Routines | Data Standardization (Z-Score), Naive Bayes, PCA |

## Theoretical Invariants

1. **Dot Product & Cosine Similarity:**
   * Dot product computes directional projection: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i$.
   * Cosine similarity normalizes magnitude using $L_2$ norm to measure angular orientation:
     $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2}$$

2. **Matrix Multiplication ($C = A \times B$):**
   * Inner dimensions must match: $(m \times k) \times (k \times n) \to (m \times n)$.
   * Triple-loop scratch implementations incur $O(n^3)$ computational complexity and cache misses.

3. **Gradient Descent Optimization:**
   * Central difference approximation for numerical verification:
     $$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$
   * First-order parameter update rule:
     $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

4. **Statistical Estimators:**
   * Bessel's correction applied for unbiased sample variance ($n - 1$ denominator):
     $$s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
   * Gaussian Normal Distribution Probability Density Function:
     $$P(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right)$$