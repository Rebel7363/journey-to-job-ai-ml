import time


# Approach 1: Quick one-liner using sorting
def is_anagram_sort(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    return sorted(s) == sorted(t)


# Approach 2: Hash map (manual frequency counter)
def is_anagram_map(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    counts = {}

    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1

    for ch in t:
        if ch not in counts or counts[ch] == 0:
            return False
        counts[ch] -= 1

    return True


# Approach 3: 26-element array (fastest for lowercase english letters)
def is_anagram_array(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    freq = [0] * 26
    for i in range(len(s)):
        freq[ord(s[i]) - ord("a")] += 1
        freq[ord(t[i]) - ord("a")] -= 1

    for val in freq:
        if val != 0:
            return False

    return True


if __name__ == "__main__":
    # Quick sanity check
    cases = [
        ("anagram", "nagaram"),
        ("rat", "car"),
        ("listen", "silent"),
    ]

    print("--- Test Cases ---")
    for s, t in cases:
        print(f"s = '{s}', t = '{t}' -> {is_anagram_map(s, t)}")

    # Simple speed test on a long string
    long_s = "racecar" * 50_000
    long_t = "carrace" * 50_000

    t0 = time.time()
    is_anagram_sort(long_s, long_t)
    t_sort = time.time() - t0

    t0 = time.time()
    is_anagram_map(long_s, long_t)
    t_map = time.time() - t0

    t0 = time.time()
    is_anagram_array(long_s, long_t)
    t_arr = time.time() - t0

    print("\n--- Benchmark (350k chars) ---")
    print(f"Sorting: {t_sort:.4f}s")
    print(f"HashMap: {t_map:.4f}s")
    print(f"Array:   {t_arr:.4f}s")