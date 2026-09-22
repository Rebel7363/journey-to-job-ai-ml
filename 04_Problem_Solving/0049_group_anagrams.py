from collections import defaultdict
import time


# Approach 1: Sorted string as key
# Time: O(n * k log k) where n = len(strs), k = max string length
def group_anagrams_sort(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)

    for word in strs:
        sorted_key = "".join(sorted(word))
        groups[sorted_key].append(word)

    return list(groups.values())


# Approach 2: 26-character count tuple as key (faster for long strings)
# Time: O(n * k) | Space: O(n * k)
def group_anagrams_count(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)

    for word in strs:
        count = [0] * 26
        for ch in word:
            count[ord(ch) - ord("a")] += 1

        # tuple is hashable, list is not
        groups[tuple(count)].append(word)

    return list(groups.values())


if __name__ == "__main__":
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]

    print("--- Test Case ---")
    print("Input:", words)
    print("Output (Sorted Key):", group_anagrams_sort(words))
    print("Output (Count Key): ", group_anagrams_count(words))

    # Benchmark with a large list of words
    large_input = words * 15_000  # 90,000 words

    t0 = time.time()
    group_anagrams_sort(large_input)
    t_sort = time.time() - t0

    t0 = time.time()
    group_anagrams_count(large_input)
    t_count = time.time() - t0

    print("\n--- Benchmark (90k words) ---")
    print(f"Sorted string key: {t_sort:.4f}s")
    print(f"Count tuple key:   {t_count:.4f}s")