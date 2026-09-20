"""
Module 03: Algorithms & Patterns
Topic: Two Pointers and Sliding Window Algorithmic Paradigms
"""


def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    """
    Two Pointers Technique: O(n) Time | O(1) Space
    Sorted array me pair sum dhundhne ke liye pointers left aur right se converge hote hain.
    """
    left = 0
    right = len(numbers) - 1

    while left < right:
        curr_sum = numbers[left] + numbers[right]
        if curr_sum == target:
            return [left, right]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1

    return []


def max_subarray_sum_fixed_window(arr: list[int], k: int) -> int:
    """
    Sliding Window Technique (Fixed Size): O(n) Time | O(1) Space
    Window ko aage slide karte waqt naye element ko add aur purane ko subtract karta hai.
    """
    n = len(arr)
    if n < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum


def remove_duplicates_in_place(nums: list[int]) -> int:
    """
    Fast & Slow Pointers: O(n) Time | O(1) Auxiliary Space
    Sorted array se duplicates ko in-place remove karta hai.
    """
    if not nums:
        return 0

    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1


if __name__ == "__main__":
    print("=" * 60)
    print("1. TWO POINTERS: TARGET PAIR SEARCH")
    print("=" * 60)
    sorted_data = [2, 7, 11, 15, 19, 23]
    target_val = 26
    indices = two_sum_sorted(sorted_data, target_val)
    print(f"Array: {sorted_data} | Target: {target_val}")
    print(f"Indices: {indices} (Values: {sorted_data[indices[0]]} + {sorted_data[indices[1]]})")

    print("\n" + "=" * 60)
    print("2. SLIDING WINDOW: MAXIMUM SUBARRAY SUM OF SIZE K")
    print("=" * 60)
    stream_metrics = [100, 200, 300, 400, 250, 600, 150]
    window_k = 3
    print(f"Metrics: {stream_metrics} | k: {window_k}")
    print(f"Max Sum: {max_subarray_sum_fixed_window(stream_metrics, window_k)}")

    print("\n" + "=" * 60)
    print("3. IN-PLACE DEDUPLICATION (SLOW/FAST POINTERS)")
    print("=" * 60)
    duplicate_arr = [1, 1, 2, 2, 2, 3, 4, 4, 5]
    unique_len = remove_duplicates_in_place(duplicate_arr)
    print(f"Unique Length: {unique_len}")
    print(f"Deduplicated Prefix: {duplicate_arr[:unique_len]}")