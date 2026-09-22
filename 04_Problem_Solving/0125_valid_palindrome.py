import time


# Approach 1: Filter alphanumeric chars then reverse
# Time: O(n) | Space: O(n) for new string
def is_palindrome_simple(s: str) -> bool:
    cleaned = [ch.lower() for ch in s if ch.isalnum()]
    return cleaned == cleaned[::-1]


# Approach 2: In-place two pointers (skips non-alphanumeric on the fly)
# Time: O(n) | Space: O(1) auxiliary
def is_palindrome_two_pointers(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True


if __name__ == "__main__":
    cases = [
        "A man, a plan, a canal: Panama",
        "race a car",
        " ",
        "0P",
    ]

    print("--- Test Cases ---")
    for text in cases:
        print(f"'{text}' -> {is_palindrome_two_pointers(text)}")

    # Benchmark memory/time efficiency on a massive string
    big_s = "A man, a plan, a canal: Panama! " * 50_000

    t0 = time.time()
    is_palindrome_simple(big_s)
    t_simple = time.time() - t0

    t0 = time.time()
    is_palindrome_two_pointers(big_s)
    t_ptr = time.time() - t0

    print("\n--- Benchmark (1.6M characters) ---")
    print(f"Filter + Reverse (O(n) space): {t_simple:.4f}s")
    print(f"Two Pointers     (O(1) space): {t_ptr:.4f}s")