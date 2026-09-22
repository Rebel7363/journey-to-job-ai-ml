import time


# Standard Iterative Binary Search: O(log n) Time | O(1) Space
def binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        # Prevents integer overflow in languages with fixed-width integers
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# Linear Scan Baseline: O(n) Time | O(1) Space
def linear_search(nums: list[int], target: int) -> int:
    for i, val in enumerate(nums):
        if val == target:
            return i
    return -1


if __name__ == "__main__":
    cases = [
        ([-1, 0, 3, 5, 9, 12], 9),
        ([-1, 0, 3, 5, 9, 12], 2),
        ([5], 5),
        ([5], -5),
    ]

    print("--- Test Cases ---")
    for arr, tgt in cases:
        idx = binary_search(arr, tgt)
        print(f"Nums: {arr}, Target: {tgt} -> Found at index: {idx}")

    # Benchmark: Search last element in 1,000,000 items
    large_sorted = list(range(1_000_000))
    target_val = 999_999

    t0 = time.time()
    linear_search(large_sorted, target_val)
    t_linear = time.time() - t0

    t0 = time.time()
    binary_search(large_sorted, target_val)
    t_binary = time.time() - t0

    print("\n--- Benchmark (1,000,000 items) ---")
    print(f"Linear Search O(n):     {t_linear:.6f}s")
    print(f"Binary Search O(log n): {t_binary:.6f}s")
    print(f"Speedup: ~{t_linear / max(t_binary, 1e-9):.1f}x")