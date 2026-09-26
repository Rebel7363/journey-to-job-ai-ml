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
     $$\mathbf{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^TyYour CART scratch implementation is running cleanly and showing solid test results:

* **Script:** `06_Classical_ML/05_decision_tree.py`[cite: 1]
* **Dataset Split:** 64 train samples / 16 test samples (80/20 split)[cite: 1]
* **Root Split:** Feature index `1` at threshold `0.0646`[cite: 1]
* **Performance:** **93.75%** test accuracy (15/16 correct predictions)[cite: 1]

Are you looking to add tree pruning, compare this with scikit-learn's `DecisionTreeClassifier`, or move on to building an ensemble like Random Forest next?