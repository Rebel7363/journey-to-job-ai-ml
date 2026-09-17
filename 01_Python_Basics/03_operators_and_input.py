"""
Day 2: Operators & Dynamic User Input
-------------------------------------
Demonstrating:
1. Dynamic terminal input via input()
2. Explicit input casting (strings to numerical values)
3. Arithmetic & Modulo operations
4. Comparison & Logical operations
5. Membership (in) & Identity (is) operators
"""

print("=" * 50)
print("1. USER INPUT & ARITHMETIC OPERATORS")
print("=" * 50)

# input() hamesha data string (str) format me leta hai
raw_num1 = input("Enter first number (e.g., 10): ") or "10"
raw_num2 = input("Enter second number (e.g., 3): ") or "3"

# Explicit conversion to float for numerical accuracy
a = float(raw_num1)
b = float(raw_num2)

print(f"\nOperands: a = {a}, b = {b}")
print(f"Addition (a + b): {a + b}")
print(f"Subtraction (a - b): {a - b}")
print(f"Multiplication (a * b): {a * b}")
print(f"Float Division (a / b): {a / b:.2f}")
print(f"Floor Division (a // b): {a // b} (Removes decimal)")
print(f"Modulo / Remainder (a % b): {a % b}")
print(f"Exponentiation (a ** b): {a ** b}")

print("\n" + "=" * 50)
print("2. COMPARISON & LOGICAL OPERATORS")
print("=" * 50)

print(f"Is a equal to b? (a == b): {a == b}")
print(f"Is a strictly greater than b? (a > b): {a > b}")

# Logical operations (and, or, not)
is_positive = (a > 0) and (b > 0)
has_negative = (a < 0) or (b < 0)
print(f"Both positive? ((a > 0) and (b > 0)): {is_positive}")
print(f"Any negative? ((a < 0) or (b < 0)): {has_negative}")
print(f"Negation check (not is_positive): {not is_positive}")

print("\n" + "=" * 50)
print("3. MEMBERSHIP & IDENTITY OPERATORS")
print("=" * 50)

# Membership testing: 'in' / 'not in'
tech_stack = ["python", "numpy", "pandas", "pytorch"]
search_query = "python"
print(f"Is '{search_query}' in tech_stack? {'python' in tech_stack}")

# Identity testing: 'is' vs '==' (checks memory reference vs value)
list_one = [1, 2, 3]
list_two = [1, 2, 3]
print(f"Value Equality (list_one == list_two): {list_one == list_two}")
print(f"Memory Identity (list_one is list_two): {list_one is list_two} (Separate memory locations)")

print("\nExecution complete: All operator behaviors mapped.")6