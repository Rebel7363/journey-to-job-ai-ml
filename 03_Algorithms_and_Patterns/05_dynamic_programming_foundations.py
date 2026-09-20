"""
Module 03: Algorithms & Patterns
Topic: Dynamic Programming Foundations (Memoization vs Tabulation, 0/1 Knapsack, and LCS)
"""

import time


# =====================================================================
# 1. FIBONACCI: TOP-DOWN (MEMOIZATION) VS BOTTOM-UP (TABULATION)
# =====================================================================
def fib_memoization(n: int, memo: dict = None) -> int:
    """Top-Down DP with Memoization: O(n) Time | O(n) Call Stack + Map Space."""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n

    memo[n] = fib_memoization(n - 1, memo) + fib_memoization(n - 2, memo)
    return memo[n]


def fib_tabulation_optimized(n: int) -> int:
    """Bottom-Up Space-Optimized DP: O(n) Time | O(1) Auxiliary Space."""
    if n <= 1:
        return n

    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1


# =====================================================================
# 2. 0/1 KNAPSACK PROBLEM (TABULATION MATRIX)
# =====================================================================
def zero_one_knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Classic 0/1 Knapsack: O(n * W) Time | O(n * W) Space
    DP Table: dp[i][w] stores max value with first i items and capacity w.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for cap in range(1, capacity + 1):
            if w <= cap:
                dp[i][cap] = max(dp[i - 1][cap], v + dp[i - 1][cap - w])
            else:
                dp[i][cap] = dp[i - 1][cap]

    return dp[n][capacity]


# =====================================================================
# 3. LONGEST COMMON SUBSEQUENCE (NLP / SEQUENCE ALIGNMENT)
# =====================================================================
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    LCS: O(m * n) Time | O(m * n) Space
    Core pattern used in DNA alignment, diff tools (git diff), and NLP similarity.
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


if __name__ == "__main__":
    print("=" * 65)
    print("1. TOP-DOWN MEMOIZATION VS BOTTOM-UP TABULATION")
    print("=" * 65)
    target_n = 50
    t0 = time.perf_counter()
    memo_res = fib_memoization(target_n)
    t_memo = time.perf_counter() - t0

    t0 = time.perf_counter()
    tab_res = fib_tabulation_optimized(target_n)
    t_tab = time.perf_counter() - t0

    print(f"Fib({target_n}) Memoization: {memo_res} (Elapsed: {t_memo:.6f}s)")
    print(f"Fib({target_n}) Tabulation:  {tab_res} (Elapsed: {t_tab:.6f}s)")

    print("\n" + "=" * 65)
    print("2. 0/1 KNAPSACK PROBLEM (INTEGER PROGRAMMING)")
    print("=" * 65)
    item_weights = [2, 3, 4, 5]
    item_values = [3, 4, 5, 6]
    max_bag_cap = 5
    max_attainable_val = zero_one_knapsack(item_weights, item_values, max_bag_cap)
    print(f"Weights: {item_weights} | Values: {item_values}")
    print(f"Capacity: {max_bag_cap} -> Max Optimal Value: {max_attainable_val}")

    print("\n" + "=" * 65)
    print("3. LONGEST COMMON SUBSEQUENCE (NLP / DIFF SIMILARITY)")
    print("=" * 65)
    seq_a = "TRANSFORMER"
    seq_b = "PERFORMER"
    lcs_len = longest_common_subsequence(seq_a, seq_b)
    print(f"Sequence 1: {seq_a}")
    print(f"Sequence 2: {seq_b}")
    print(f"Length of Longest Common Subsequence: {lcs_len}")