import time


# Approach 1: Brute force checking all pairs
# Time: O(n^2) | Space: O(1)
def max_profit_brute_force(prices: list[int]) -> int:
    max_profit = 0
    n = len(prices)

    for i in range(n):
        for j in range(i + 1, n):
            profit = prices[j] - prices[i]
            if profit > max_profit:
                max_profit = profit

    return max_profit


# Approach 2: One-pass tracking minimum buy price
# Time: O(n) | Space: O(1)
def max_profit_one_pass(prices: list[int]) -> int:
    min_price = float("inf")
    max_profit = 0

    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price

    return max_profit


if __name__ == "__main__":
    cases = [
        [7, 1, 5, 3, 6, 4],
        [7, 6, 4, 3, 1],
        [2, 4, 1],
    ]

    print("--- Test Cases ---")
    for p in cases:
        print(f"Prices: {p} -> Max Profit: {max_profit_one_pass(p)}")

    # Benchmark on 10,000 price ticks
    simulated_prices = list(range(10_000, 0, -1)) + [100, 500, 2000]

    t0 = time.time()
    max_profit_brute_force(simulated_prices)
    t_bf = time.time() - t0

    t0 = time.time()
    max_profit_one_pass(simulated_prices)
    t_opt = time.time() - t0

    print("\n--- Benchmark (10k items) ---")
    print(f"Brute Force (O(n^2)): {t_bf:.4f}s")
    print(f"One-Pass    (O(n)):   {t_opt:.6f}s")
    print(f"Speedup: ~{t_bf / t_opt:.1f}x")