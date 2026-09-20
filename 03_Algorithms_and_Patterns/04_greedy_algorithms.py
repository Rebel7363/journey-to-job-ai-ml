"""
Module 03: Algorithms & Patterns
Topic: Greedy Choice Property, Interval Scheduling, and Fractional Knapsack
"""


def interval_scheduling_max_tasks(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Greedy Interval Scheduling: O(n log n) Time | O(n) Space
    Always picks the task with the earliest finish time to maximize free time.
    """
    if not intervals:
        return []

    # Sort by end time
    sorted_intervals = sorted(intervals, key=lambda x: x[1])
    selected = [sorted_intervals[0]]
    last_end = sorted_intervals[0][1]

    for start, end in sorted_intervals[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected


def fractional_knapsack(capacity: float, items: list[dict]) -> float:
    """
    Fractional Knapsack (Greedy): O(n log n) Time | O(1) Auxiliary Space
    Selects items based on maximum value-to-weight ratio.
    """
    # Sort descending by value / weight
    items_sorted = sorted(items, key=lambda x: x["value"] / x["weight"], reverse=True)

    total_value = 0.0
    remaining_cap = capacity

    for item in items_sorted:
        if remaining_cap == 0:
            break

        weight = item["weight"]
        value = item["value"]

        if weight <= remaining_cap:
            total_value += value
            remaining_cap -= weight
        else:
            # Take fraction
            fraction = remaining_cap / weight
            total_value += value * fraction
            remaining_cap = 0

    return total_value


def min_coins_change_greedy(coins: list[int], amount: int) -> list[int]:
    """
    Greedy Coin Change (Works for canonical coin systems like INR/USD).
    Time: O(n log n) | Space: O(k)
    """
    coins_sorted = sorted(coins, reverse=True)
    change = []

    for coin in coins_sorted:
        while amount >= coin:
            amount -= coin
            change.append(coin)

    return change if amount == 0 else []


if __name__ == "__main__":
    print("=" * 65)
    print("1. GREEDY INTERVAL SCHEDULING (CPU TASK ALLOCATION)")
    print("=" * 65)
    tasks = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
    scheduled = interval_scheduling_max_tasks(tasks)
    print(f"Total Submitted Tasks: {len(tasks)}")
    print(f"Max Non-Overlapping Scheduled Tasks: {len(scheduled)}")
    print(f"Scheduled (Start, End): {scheduled}")

    print("\n" + "=" * 65)
    print("2. FRACTIONAL KNAPSACK (VALUE / WEIGHT RATIO)")
    print("=" * 65)
    knapsack_items = [
        {"name": "Gold_Dust", "value": 280, "weight": 40},
        {"name": "Silver", "value": 100, "weight": 10},
        {"name": "Copper", "value": 120, "weight": 20},
        {"name": "Platinum", "value": 120, "weight": 24},
    ]
    max_capacity = 60
    max_val = fractional_knapsack(max_capacity, knapsack_items)
    print(f"Max Value for Capacity {max_capacity}: {max_val:.2f}")

    print("\n" + "=" * 65)
    print("3. GREEDY COIN CHANGE (CANONICAL DENOMINATIONS)")
    print("=" * 65)
    denominations = [1, 2, 5, 10, 20, 50, 100, 500]
    target_amount = 788
    coins_given = min_coins_change_greedy(denominations, target_amount)
    print(f"Target Amount: ₹{target_amount}")
    print(f"Total Coins/Notes: {len(coins_given)}")
    print(f"Breakdown: {coins_given}")