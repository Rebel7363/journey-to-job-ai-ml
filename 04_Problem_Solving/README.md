# Module 04: Problem Solving & LeetCode Benchmarks

This module focuses on high-frequency algorithmic problem-solving patterns. Each implementation includes complexity analysis, invariant guarantees, and execution benchmarks comparing brute-force baselines against optimal algorithms.

## Problem Index

| ID | Problem Title | Pattern / Technique | Time Complexity | Space Complexity | Solution File |
|:---|:---|:---|:---:|:---:|:---|
| 0001 | Two Sum | Hash Map Complement Lookup | $O(n)$ | $O(n)$ | [`0001_two_sum.py`](./0001_two_sum.py) |
| 0015 | 3Sum | Two Pointers + Sorting Deduplication | $O(n^2)$ | $O(1)$ | [`0015_three_sum.py`](./0015_three_sum.py) |
| 0020 | Valid Parentheses | LIFO Stack Delimiter Matching | $O(n)$ | $O(n)$ | [`0020_valid_parentheses.py`](./0020_valid_parentheses.py) |
| 0049 | Group Anagrams | Character Frequency Hashing | $O(n \cdot k)$ | $O(n \cdot k)$ | [`0049_group_anagrams.py`](./0049_group_anagrams.py) |
| 0121 | Best Time to Buy and Sell Stock | Single-Pass Running Minimum Tracking | $O(n)$ | $O(1)$ | [`0121_best_time_to_buy_and_sell_stock.py`](./0121_best_time_to_buy_and_sell_stock.py) |
| 0125 | Valid Palindrome | In-Place Two Pointers Convergence | $O(n)$ | $O(1)$ | [`0125_valid_palindrome.py`](./0125_valid_palindrome.py) |
| 0217 | Contains Duplicate | Hash Set Early Exit Invariant | $O(n)$ | $O(n)$ | [`0217_contains_duplicate.py`](./0217_contains_duplicate.py) |
| 0242 | Valid Anagram | 26-Element Fixed Frequency Array | $O(n)$ | $O(1)$ | [`0242_valid_anagram.py`](./0242_valid_anagram.py) |
| 0704 | Binary Search | Overflow-Safe Halving Invariant | $O(\log n)$ | $O(1)$ | [`0704_binary_search.py`](./0704_binary_search.py) |

## Core Patterns Covered

1. **Hash Maps & Sets:** Used for constant-time $O(1)$ membership checks, complementary value matching, and anagram grouping without sorting overhead.
2. **Two Pointers:** Moving inward on sorted sequences to eliminate nested combinatorial loops, reducing search spaces from $O(n^3)$ to $O(n^2)$ or $O(n^2)$ to $O(n)$.
3. **Running Invariants (Greedy):** Preserving local extrema (running minimum/maximum) in a single linear scan.
4. **Delimited Stacks:** Validating nested balanced structures using Last-In-First-Out retrieval.
5. **Divide & Conquer Search:** Halving search intervals monotonically using midpoint invariants.