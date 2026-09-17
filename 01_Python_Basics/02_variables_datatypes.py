"""
Day 2: Variables, Data Types & Type Casting
-------------------------------------------
Demonstrating:
1. Dynamic typing & memory references
2. Primitive data types (int, float, str, bool)
3. Type inspection using type()
4. Explicit Type Casting & error handling
"""

print("=" * 50)
print("1. PRIMITIVE DATA TYPES & TYPE INSPECTION")
print("=" * 50)

# Variables define karte hain (Python dynamically type assign karta hai)
user_name = "Raj"             # str (String)
experience_years = 1          # int (Integer)
accuracy_score = 98.6         # float (Floating point)
is_learning_ai = True         # bool (Boolean)

print(f"user_name: {user_name} | Type: {type(user_name)}")
print(f"experience_years: {experience_years} | Type: {type(experience_years)}")
print(f"accuracy_score: {accuracy_score} | Type: {type(accuracy_score)}")
print(f"is_learning_ai: {is_learning_ai} | Type: {type(is_learning_ai)}")

print("\n" + "=" * 50)
print("2. DYNAMIC TYPING IN ACTION")
print("=" * 50)

# Ek hi variable alag type ka data hold kar sakta hai
data_holder = 100
print(f"Initial: {data_holder} -> {type(data_holder)}")

data_holder = "Now I am text"
print(f"Reassigned: '{data_holder}' -> {type(data_holder)}")

print("\n" + "=" * 50)
print("3. EXPLICIT TYPE CASTING")
print("=" * 50)

# String se Number conversion (Common in user inputs/APIs)
str_number = "50"
converted_int = int(str_number)
print(f"String to Int addition: {converted_int + 25}")

# Float to Int (Truncation check)
pi_value = 3.99
truncated_int = int(pi_value)
print(f"Float {pi_value} cast to int: {truncated_int} (Decimal drops)")

# Boolean casting rules
print(f"bool(1): {bool(1)}")       # Non-zero number -> True
print(f"bool(0): {bool(0)}")       # Zero -> False
print(f"bool(''): {bool('')}")     # Empty string -> False
print(f"bool('AI'): {bool('AI')}") # Non-empty string -> True

print("\nExecution complete: All types validated.")