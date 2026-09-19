"""
Module 02: Data Structures
Topic: Stacks, Queues, Deque Architecture, and Monotonic Stack Pattern
"""

from collections import deque
import time


def benchmark_list_vs_deque_queue(operations: int = 60_000):
    """
    Demonstrates why using Python list as a queue is an anti-pattern.
    list.pop(0) costs O(n) due to contiguous memory shifts.
    collections.deque.popleft() costs true O(1) via block pointer adjustments.
    """
    print("=" * 65)
    print(f"1. QUEUE PERFORMANCE BENCHMARK ({operations:,} OPERATIONS)")
    print("=" * 65)

    # 1. Anti-Pattern: List as Queue
    list_q = list(range(operations))
    start_list = time.perf_counter()
    while list_q:
        list_q.pop(0)
    list_time = time.perf_counter() - start_list

    # 2. Optimized: collections.deque as Queue
    deque_q = deque(range(operations))
    start_deque = time.perf_counter()
    while deque_q:
        deque_q.popleft()
    deque_time = time.perf_counter() - start_deque

    print(f"list.pop(0) [O(n) shifts]:       {list_time:.4f} seconds")
    print(f"deque.popleft() [O(1) pointers]:  {deque_time:.4f} seconds")
    print(f"Result: deque is ~{list_time / deque_time:.1f}x faster for queue pops\n")


class LIFOStack:
    """Standard Stack implementation using Python dynamic array."""

    def __init__(self):
        self._container = []

    def push(self, item):
        self._container.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._container.pop()

    def peek(self):
        return self._container[-1] if not self.is_empty() else None

    def is_empty(self) -> bool:
        return len(self._container) == 0

    def size(self) -> int:
        return len(self._container)


def is_valid_parentheses(expression: str) -> bool:
    """
    Classic Stack application: Syntax validation in compilers/parsers.
    Time: O(n) | Space: O(n)
    """
    brackets = {")": "(", "}": "{", "]": "["}
    stack = LIFOStack()

    for char in expression:
        if char in brackets.values():
            stack.push(char)
        elif char in brackets:
            if stack.is_empty() or stack.pop() != brackets[char]:
                return False

    return stack.is_empty()


def next_greater_element(temperatures: list[int]) -> list[int]:
    """
    Monotonic Decreasing Stack Pattern (Core DSA / Time-Series / AI Feature Eng).
    Finds the number of steps to wait for a higher value.
    Time: O(n) | Space: O(n)
    """
    n = len(temperatures)
    days_to_wait = [0] * n
    index_stack = []  # Stores indices in monotonic decreasing order of values

    for curr_idx, curr_val in enumerate(temperatures):
        while index_stack and temperatures[index_stack[-1]] < curr_val:
            prev_idx = index_stack.pop()
            days_to_wait[prev_idx] = curr_idx - prev_idx
        index_stack.append(curr_idx)

    return days_to_wait


if __name__ == "__main__":
    benchmark_list_vs_deque_queue(50_000)

    print("=" * 65)
    print("2. STACK PARSING: SYNTAX VALIDATION")
    print("=" * 65)
    test_expressions = ["{ [ a * (b + c) ] }", "{ [ ( ] ) }", "(())["]
    for expr in test_expressions:
        print(f"Expression: {expr:<20} -> Valid: {is_valid_parentheses(expr)}")

    print("\n" + "=" * 65)
    print("3. MONOTONIC STACK: TIME-SERIES / NEXT GREATER VALUE")
    print("=" * 65)
    sample_series = [73, 74, 75, 71, 69, 72, 76, 73]
    wait_steps = next_greater_element(sample_series)
    print(f"Values:     {sample_series}")
    print(f"Wait Steps: {wait_steps}")