"""
Module 02: Data Structures
Topic: Dictionaries & Sets Internals, Hash Table Architecture, and Benchmarks
"""

import sys
import timeit


def inspect_hash_and_indexing():
    """Demonstrates how hash values map to table indices."""
    print("=" * 65)
    print("1. HASH FUNCTION & BUCKET MAPPING")
    print("=" * 65)

    keys = ["model_name", "learning_rate", "batch_size", "epochs", "optimizer"]
    table_size = 8  # Initial default capacity in CPython

    for k in keys:
        h = hash(k)
        idx = h & (table_size - 1)  # Bitwise modulo: hash % table_size
        print(f"Key: {k:<15} | Hash: {h:<22} | Bucket Index: {idx}")


def inspect_dict_growth_and_memory():
    """Tracks memory reallocation of a dynamic dictionary."""
    print("\n" + "=" * 65)
    print("2. DICTIONARY DYNAMIC RESIZING (CPython Compact Dict)")
    print("=" * 65)

    data_map = {}
    prev_bytes = sys.getsizeof(data_map)
    print(f"Empty dict initial memory: {prev_bytes} bytes\n")

    print(f"{'Length':<10}{'Capacity/Bytes':<20}{'Status'}")
    print("-" * 45)

    for i in range(25):
        data_map[f"param_{i}"] = i * 2
        curr_bytes = sys.getsizeof(data_map)
        if curr_bytes != prev_bytes:
            delta = curr_bytes - prev_bytes
            print(f"{len(data_map):<10}{curr_bytes:<20}Resized (+{delta} bytes)")
            prev_bytes = curr_bytes


def compare_lookup_complexity():
    """Benchmarks O(1) Set/Dict lookup vs O(n) List lookup."""
    print("\n" + "=" * 65)
    print("3. LOOKUP SPEED: SET / DICT O(1) VS LIST O(n)")
    print("=" * 65)

    size = 100_000
    target = 99_999  # Worst-case search target

    test_list = list(range(size))
    test_set = set(range(size))
    test_dict = {i: True for i in range(size)}

    iterations = 1_000

    list_time = timeit.timeit(lambda: target in test_list, number=iterations)
    set_time = timeit.timeit(lambda: target in test_set, number=iterations)
    dict_time = timeit.timeit(lambda: target in test_dict, number=iterations)

    print(f"List search ({iterations} runs): {list_time:.5f} seconds  [O(n)]")
    print(f"Set search  ({iterations} runs): {set_time:.5f} seconds  [O(1)]")
    print(f"Dict search ({iterations} runs): {dict_time:.5f} seconds  [O(1)]")
    print(f"Set is ~{list_time / set_time:.1f}x faster than list for searches!")


def demonstrate_nlp_set_operations():
    """Real-world AI/NLP connection: Vocabulary building & set theory."""
    print("\n" + "=" * 65)
    print("4. AI / NLP CONNECTION: VOCABULARY SET THEORY")
    print("=" * 65)

    doc_a = "machine learning pipelines optimize tensor computations in neural networks".split()
    doc_b = "neural networks require scalable data pipelines and fast computations".split()

    vocab_a = set(doc_a)
    vocab_b = set(doc_b)

    common_tokens = vocab_a.intersection(vocab_b)
    unique_to_a = vocab_a.difference(vocab_b)
    total_vocab = vocab_a.union(vocab_b)

    # Jaccard Similarity = |A ∩ B| / |A ∪ B|
    jaccard_similarity = len(common_tokens) / len(total_vocab)

    print(f"Document A Vocab Size: {len(vocab_a)}")
    print(f"Document B Vocab Size: {len(vocab_b)}")
    print(f"Common Tokens (A ∩ B): {common_tokens}")
    print(f"Unique to Doc A (A - B): {unique_to_a}")
    print(f"Jaccard Token Similarity: {jaccard_similarity:.3f}")


if __name__ == "__main__":
    inspect_hash_and_indexing()
    inspect_dict_growth_and_memory()
    compare_lookup_complexity()
    demonstrate_nlp_set_operations()