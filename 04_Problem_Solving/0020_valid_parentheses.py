import time


# Stack-based approach: O(n) Time | O(n) Space
def is_valid_parentheses(s: str) -> bool:
    stack = []
    lookup = {")": "(", "}": "{", "]": "["}

    for char in s:
        if char in lookup:
            # Closing bracket encountered
            top = stack.pop() if stack else "#"
            if lookup[char] != top:
                return False
        else:
            # Opening bracket encountered
            stack.append(char)

    return len(stack) == 0


if __name__ == "__main__":
    cases = [
        "()",
        "()[]{}",
        "(]",
        "([)]",
        "{[]}",
        "((((",
    ]

    print("--- Test Cases ---")
    for pattern in cases:
        print(f"'{pattern}' -> {is_valid_parentheses(pattern)}")

    # Benchmark on deeply nested valid parentheses
    deep_nested = "(" * 100_000 + ")" * 100_000

    t0 = time.time()
    res = is_valid_parentheses(deep_nested)
    elapsed = time.time() - t0

    print(f"\n--- Benchmark (200k characters) ---")
    print(f"Result: {res}")
    print(f"Elapsed Time: {elapsed:.4f}s")