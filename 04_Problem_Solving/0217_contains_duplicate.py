"""
Problem: LeetCode 217 - Contains Duplicate
Difficulty: Easy
Patterns: Hash Set Lookup, Early Exit Invariant
"""

import time


def contains_duplicate_brute_force(nums: list[int]) -> bool:
    """Brute Force: O(n^2) Time | O(1) Space"""
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] == nums[j]:
                return True
    return False


def contains_duplicate_hash_set(nums: list[int]) -> bool:
    """
    Hash Set Lookup: O(n) Time | O(n) Space
    Worst-case visits every element once; returns True on first repeated hit.
    """
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


if __name__ == "__main__":
    test_cases = [
        [1, 2, 3, 1],
        [1, 2, 3, 4],
        [1, 1, 1, 3, 3, 4, 3, 2, 4, 2],
    ]

    print("=" * 60)
    print("LEETCODE 0217: CONTAINS DUPLICATE BENCHMARK")
    print("=" * 60)

    for case in test_cases:
        print(f"Array: {case}")
        print(f"Has Duplicate (Hash Set): {contains_duplicate_hash_set(case)}\n")

    # Benchmark: Worst-case where all items are unique
    large_unique_array = list(range(10_000))

    t0 = time.perf_counter()
    contains_duplicate_brute_force(large_unique_array)
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    contains_duplicate_hash_set(large_unique_array)
    t_set = time.perf_counter() - t0

    print("Large Unique Array Benchmark (10,000 items):")
    print(f"Brute Force O(n^2): {t_bf:.5f}s")
    print(f"Hash Set O(n):    {t_set:.5f}s")
    print(f"Speedup: ~{t_bf / t_set:.1f}x faster")