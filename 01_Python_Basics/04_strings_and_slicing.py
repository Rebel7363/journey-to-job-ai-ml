"""
Day 2: Strings, Indexing, Slicing & Methods
------------------------------------------
Demonstrating:
1. Positive and negative indexing
2. Slicing with [start:stop:step]
3. String reversal trick
4. Essential string methods (strip, split, replace, join)
5. Immutability & Exception Handling (Production pattern)
6. Data Preprocessing / Tokenization basics for AI
"""

sample_text = "Python Machine Learning"

print("=" * 50)
print("1. INDEXING & SLICING BASICS")
print("=" * 50)

# Direct f-string debug formatting (Modern Python 3.8+)

print(f"First character: sample_text[0] = {sample_text[0]}")
print(f"Last character:  sample_text[-1] = {sample_text[-1]}")

# Slicing: [start : stop] (stop index is excluded)

print(f"Slice [0:6]:     '{sample_text[0:6]}'")
print(f"Slice [7:]:      '{sample_text[7:]}'")

print("\n" + "=" * 50)
print("2. ADVANCED SLICING WITH STEP")
print("=" * 50)

numbers = "0123456789"
print(f"Original sequence:    {numbers}")
print(f"Every second item:    {numbers[::2]}")
print(f"Step 2 (range 1 to 8):{numbers[1:8:2]}")
print(f"Idiomatic Reversal:   '{sample_text[::-1]}'")

print("\n" + "=" * 50)
print("3. IMMUTABILITY CHECK (CRITICAL CONCEPT)")
print("=" * 50)

# Strings are immutable; in-place modification raises TypeError
try:
    sample_text[0] = "J"
except TypeError as error:
    print(f"Immutability Verified: Strings cannot be mutated directly!")
    print(f"Caught Error: {error}")

print("\n" + "=" * 50)
print("4. ML / DATA PREPROCESSING PATTERNS")
print("=" * 50)

dirty_input = "   BERT_Base_v2_model.ckpt   "
clean_name = dirty_input.strip().lower()

print(f"Raw Input:       '{dirty_input}'")
print(f"Clean & Lower:   '{clean_name}'")
print(f"Target Extension: {clean_name.endswith('.ckpt')}")
print(f"Character Count:  Letter 'e' appears {clean_name.count('e')} times")

# Splitting / Tokenization
feature_row = "epochs,learning_rate,loss_value"
features = feature_row.split(",")
print(f"Extracted Features (List): {features}")

# Reconstructing standard output
reconstructed = " | ".join(features)
print(f"Pipeline Pipeline Header:  {reconstructed}")

print("\nExecution complete: String manipulation patterns verified.")