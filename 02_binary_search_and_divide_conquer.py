"""
Module 03: Algorithms & Patterns
Topic: Binary Search Mechanics, Bound Invariants, and Merge Sort (Divide & Conquer)
"""


def binary_search_exact(nums: list[int], target: int) -> int:
    """
    Standard Iterative Binary Search: O(log n) Time | O(1) Auxiliary Space
    Finds exact target index or returns -1.
    """
    left = 0
    right = len(nums) - 1

    while left <= right:
        # Prevents integer overflow present in (left + right) // 2
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def lower_bound_search(nums: list[int], target: int) -> int:
    """
    Binary Search Lower Bound: O(log n) Time | O(1) Auxiliary Space
    Finds the first position where element is >= target (bisect_left equivalent).
    """
    left = 0
    right = len(nums)

    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


def merge_sort(arr: list[int]) -> list[int]:
    """
    Divide & Conquer: O(n log n) Time | O(n) Auxiliary Space
    Recursively splits array and merges sorted sub-arrays.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_sorted = merge_sort(arr[:mid])
    right_sorted = merge_sort(arr[mid:])

    return _merge(left_sorted, right_sorted)


def _merge(left: list[int], right: list[int]) -> list[int]:
    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


if __name__ == "__main__":
    print("=" * 65)
    print("1. EXACT BINARY SEARCH")
    print("=" * 65)
    sorted_dataset = [3, 8, 12, 17, 24, 35, 42, 59, 70, 88]
    target_item = 35
    idx = binary_search_exact(sorted_dataset, target_item)
    print(f"Data:   {sorted_dataset}")
    print(f"Target: {target_item} -> Found at index: {idx}")

    print("\n" + "=" * 65)
    print("2. BOUND SEARCH (LOWER BOUND / INSERTION POSITION)")
    print("=" * 65)
    duplicated_data = [1, 2, 4, 4, 4, 6, 7, 9]
    search_target = 4
    first_occurrence = lower_bound_search(duplicated_data, search_target)
    print(f"Data:   {duplicated_data}")
    print(f"Target: {search_target} -> First occurrence/insertion index: {first_occurrence}")

    print("\n" + "=" * 65)
    print("3. DIVIDE & CONQUER: MERGE SORT")
    print("=" * 65)
    unsorted_stream = [64, 34, 25, 12, 22, 11, 90, 5]
    sorted_stream = merge_sort(unsorted_stream)
    print(f"Unsorted: {unsorted_stream}")
    print(f"Sorted:   {sorted_stream}")