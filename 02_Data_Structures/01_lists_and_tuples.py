"""
Module 02: Data Structures
Topic: Lists & Tuples Internals, Dynamic Resizing, and Benchmarking
"""

import copy
import sys
import timeit


def inspect_list_resizing_pattern():
    """Demonstrates CPython's dynamic array overallocation mechanism."""
    print("=" * 60)
    print("1. LIST RESIZING & OVERALLOCATION PATTERN (CPython)")
    print("=" * 60)

    dynamic_list = []
    prev_bytes = sys.getsizeof(dynamic_list)
    print(f"Empty list initial memory: {prev_bytes} bytes\n")

    print(f"{'Length':<10}{'Capacity/Bytes':<20}{'Status'}")
    print("-" * 45)

    for i in range(30):
        dynamic_list.append(i)
        current_bytes = sys.getsizeof(dynamic_list)

        if current_bytes != prev_bytes:
            delta = current_bytes - prev_bytes
            print(f"{len(dynamic_list):<10}{current_bytes:<20}Reallocated (+{delta} bytes)")
            prev_bytes = current_bytes


def compare_memory_and_immutability():
    """Analyzes memory footprint and immutability contrasts."""
    print("\n" + "=" * 60)
    print("2. MEMORY FOOTPRINT: LIST VS TUPLE (10 Elements)")
    print("=" * 60)

    sample_elements = list(range(10))
    lst = list(sample_elements)
    tpl = tuple(sample_elements)

    lst_size = sys.getsizeof(lst)
    tpl_size = sys.getsizeof(tpl)

    print(f"List size:  {lst_size} bytes")
    print(f"Tuple size: {tpl_size} bytes")
    print(f"Tuple saves: {lst_size - tpl_size} bytes ({((lst_size - tpl_size) / lst_size) * 100:.1f}% less overhead)")


def compare_copy_mechanics():
    """Demonstrates Reference vs Shallow Copy vs Deep Copy."""
    print("\n" + "=" * 60)
    print("3. MUTABILITY & COPY MECHANICS")
    print("=" * 60)

    original = [[1, 2], [3, 4]]
    shallow_copied = copy.copy(original)
    deep_copied = copy.deepcopy(original)

    # Modify inner mutable element
    original[0][0] = 999

    print(f"Original:        {original}")
    print(f"Shallow Copy:    {shallow_copied}  -> Inner list reference mutated!")
    print(f"Deep Copy:       {deep_copied}  -> Completely isolated memory")


def benchmark_speed():
    """Benchmarks creation time between Lists and Tuples."""
    print("\n" + "=" * 60)
    print("4. INSTANTIATION BENCHMARK (10 Million Iterations)")
    print("=" * 60)

    list_time = timeit.timeit(stmt="[1, 2, 3, 4, 5]", number=10_000_000)
    tuple_time = timeit.timeit(stmt="(1, 2, 3, 4, 5)", number=10_000_000)

    print(f"List instantiation:  {list_time:.4f} seconds")
    print(f"Tuple instantiation: {tuple_time:.4f} seconds")
    print(f"Result: Tuples are ~{list_time / tuple_time:.2f}x faster due to fixed allocation & caching")


if __name__ == "__main__":
    inspect_list_resizing_pattern()
    compare_memory_and_immutability()
    compare_copy_mechanics()
    benchmark_speed()