import time


# Two Pointers with Sorting: O(n^2) Time | O(1) or O(n) Space (depending on sort)
def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    res = []
    n = len(nums)

    for i in range(n - 2):
        # If the smallest number is positive, sum can never be 0
        if nums[i] > 0:
            break

        # Skip duplicate values for the first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n - 1

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            if current_sum < 0:
                left += 1
            elif current_sum > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])

                # Skip duplicates for second element
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for third element
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1

    return res


if __name__ == "__main__":
    cases = [
        [-1, 0, 1, 2, -1, -4],
        [0, 1, 1],
        [0, 0, 0],
    ]

    print("--- Test Cases ---")
    for arr in cases:
        print(f"Input: {arr} -> Triplets: {three_sum(arr.copy())}")

    # Benchmark on 1,500 numbers
    large_input = (list(range(-750, 750)))
    
    t0 = time.time()
    triplets = three_sum(large_input)
    elapsed = time.time() - t0

    print(f"\n--- Benchmark (1,500 elements) ---")
    print(f"Triplets found: {len(triplets)}")
    print(f"Elapsed Time: {elapsed:.4f}s")