"""
Problem: LeetCode 1 - Two Sum
Difficulty: Easy
Patterns: Hash Map Lookup, One-Pass Invariant
"""

import time


def two_sum_brute_force(nums: list[int], target: int) -> list[int]:
    """Brute Force: O(n^2) Time | O(1) Space"""
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def two_sum_hash_map(nums: list[int], target: int) -> list[int]:
    """
    One-Pass Hash Map: O(n) Time | O(n) Space
    Key insight: complementary value = target - current_val
    """
    lookup = {}  # maps value -> index

    for idx, val in enumerate(nums):
        complement = target - val
        if complement in lookup:
            return [lookup[complement], idx]
        lookup[val] = idx

    return []


if __name__ == "__main__":
    test_cases = [
        ([2, 7, 11, 15], 9),
        ([3, 2, 4], 6),
        ([3, 3], 6),
    ]

    print("=" * 60)
    print("LEETCODE 0001: TWO SUM BENCHMARK")
    print("=" * 60)

    for nums, target in test_cases:
        ans_bf = two_sum_brute_force(nums, target)
        ans_opt = two_sum_hash_map(nums, target)
        print(f"Nums: {nums} | Target: {target}")
        print(f"Brute Force: {ans_bf} | Hash Map O(n): {ans_opt}\n")

    # Benchmark with large array
    large_nums = list(range(1, 10_000))
    large_target = 19_997  # Last two elements (9998 + 9999)

    t0 = time.perf_counter()
    two_sum_brute_force(large_nums, large_target)
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    two_sum_hash_map(large_nums, large_target)
    t_opt = time.perf_counter() - t0

    print(f"Large Array Benchmark (10k items):")
    print(f"Brute Force O(n^2): {t_bf:.5f}s")
    print(f"Hash Map O(n):    {t_opt:.5f}s")
    print(f"Optimization: ~{t_bf / t_opt:.1f}x speedup")