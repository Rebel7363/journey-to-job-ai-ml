# Module 06: Classical Machine Learning Algorithms from Scratch

This module implements foundational supervised and unsupervised classical machine learning algorithms from first principles using pure Python and NumPy—without using scikit-learn for model mechanics.

## Implementations & Complexity

| Algorithm | Script | Paradigm | Training Complexity | Inference Complexity | Optimization / Core Metric |
|:---|:---|:---:|:---:|:---:|:---|
| **Linear Regression** | [`01_linear_regression.py`](./01_linear_regression.py) | Supervised Regression | $O(N \cdot d \cdot \text{iters})$ or $O(d^3)$ | $O(d)$ | Mean Squared Error (MSE), Normal Equation |
| **Logistic Regression** | [`02_logistic_regression.py`](./02_logistic_regression.py) | Supervised Classification | $O(N \cdot d \cdot \text{iters})$ | $O(d)$ | Binary Cross-Entropy Loss, Sigmoid |
| **K-Nearest Neighbors** | [`03_knn_classifier.py`](./03_knn_classifier.py) | Supervised Classification | $O(1)$ (Lazy learner) | $O(N \cdot d + N \log k)$ | Euclidean Distance Metric |
| **K-Means Clustering** | [`04_kmeans_clustering.py`](./04_kmeans_clustering.py) | Unsupervised Clustering | $O(N \cdot k \cdot d \cdot \text{iters})$ | $O(k \cdot d)$ | Within-Cluster Sum of Squares (Inertia) |
| **Decision Tree (CART)** | [`05_decision_tree.py`](./05_decision_tree.py) | Supervised Classification | $O(d \cdot N \log N \cdot \text{depth})$ | $O(\text{depth})$ | Gini Impurity, Recursive Binary Splitting |

## Core Mathematical Formulations

1. **Linear Regression:**
   * Hypothesis: $\hat{y} = \mathbf{X}\mathbf{w} + b$
   * Closed-form Analytical Solution (Normal Equation):
     $$\mathbf{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
   * Gradient updates: $\frac{\partial L}{\partial \mathbf{w}} = \frac{2}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$

2. **Logistic Regression:**
   * Sigmoid Activation: $\sigma(z) = \frac{1}{1 + e^{-z}}$
   * Binary Cross-Entropy Loss:
     $$L(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

3. **K-Nearest Neighbors (KNN):**
   * Euclidean Distance Metric:
     $$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}$$
   * Prediction: Mode of $k$ smallest distances.

4. **K-Means Clustering:**
   * Objective Function (Within-Cluster Sum of Squares / Inertia):
     $$J = \sum_{k=1}^{K} \sum_{i \in S_k} \Vert{}\mathbf{x}_i - \mathbf{\mu}_k\Vert{}^2$$
   * Centroid Update: $\mathbf{\mu}_k = \frac{1}{\vert{}S_k\vert{}} \sum_{i \in S_k} \mathbf{x}_i$

5. **Decision Tree (CART):**
   * Gini Impurity:
     $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
   * Split Evaluation: Maximize impurity decrease across binary splits.