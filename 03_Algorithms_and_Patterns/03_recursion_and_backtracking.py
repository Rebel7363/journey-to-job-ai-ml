"""
Module 03: Algorithms & Patterns
Topic: Recursion Call Stack Internals, Backtracking, and Combinatorial Search
"""

import sys


def inspect_recursion_depth(n: int, max_depth: int = 5):
    """Demonstrates CPython call stack frames and frame unwinding."""
    indent = "  " * (max_depth - n)
    print(f"{indent}-> Entering Frame: n = {n}")

    if n <= 1:
        print(f"{indent}** Base Case Reached: n = {n} **")
        return 1

    result = n * inspect_recursion_depth(n - 1, max_depth)
    print(f"{indent}<- Unwinding Frame: n = {n} | Returning: {result}")
    return result


def generate_subsets_backtracking(nums: list[int]) -> list[list[int]]:
    """
    Combinatorial Backtracking (Power Set): O(2^n) Time | O(n) Recursion Space
    Decision tree: Har element ke liye do choices (Include / Exclude).
    """
    subsets = []
    current_path = []

    def _backtrack(start_index: int):
        # Har recursive call ka state ek valid subset represent karta hai
        subsets.append(current_path.copy())

        for i in range(start_index, len(nums)):
            # 1. Choose
            current_path.append(nums[i])
            # 2. Explore
            _backtrack(i + 1)
            # 3. Un-choose (Backtrack step)
            current_path.pop()

    _backtrack(0)
    return subsets


def solve_n_queens_count(n: int) -> int:
    """
    Backtracking Constraint Satisfaction: N-Queens problem solutions count.
    Uses hash sets for O(1) diagonal and column conflict checks.
    """
    cols = set()
    diag1 = set()  # (r - c)
    diag2 = set()  # (r + c)

    def _backtrack(row: int) -> int:
        if row == n:
            return 1

        solutions = 0
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # Place queen
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            solutions += _backtrack(row + 1)

            # Backtrack
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

        return solutions

    return _backtrack(0)


if __name__ == "__main__":
    print("=" * 65)
    print("1. RECURSION CALL STACK & FRAME UNWINDING")
    print("=" * 65)
    print(f"CPython Default Recursion Limit: {sys.getrecursionlimit()}")
    factorial_val = inspect_recursion_depth(4, 4)
    print(f"Final Factorial Output: {factorial_val}")

    print("\n" + "=" * 65)
    print("2. COMBINATORIAL SEARCH: SUBSETS GENERATION")
    print("=" * 65)
    source_set = [1, 2, 3]
    all_subsets = generate_subsets_backtracking(source_set)
    print(f"Elements: {source_set}")
    print(f"Total Subsets Generated: {len(all_subsets)}")
    print(f"Subsets: {all_subsets}")

    print("\n" + "=" * 65)
    print("3. CONSTRAINT SATISFACTION: N-QUEENS COUNT")
    print("=" * 65)
    board_dim = 4
    total_placements = solve_n_queens_count(board_dim)
    print(f"Board Dimensions: {board_dim}x{board_dim}")
    print(f"Valid Non-Attacking Solutions: {total_placements}")