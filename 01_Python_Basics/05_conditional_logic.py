"""
Day 3: Conditional Logic & Branching
------------------------------------
Demonstrating:
1. Classic if-elif-else branching
2. Ternary operator (inline conditional)
3. Logical chaining with short-circuit evaluation
4. Structural Pattern Matching (match-case) for clean state handling
"""

print("=" * 50)
print("1. CLASSIC IF-ELIF-ELSE BRANCHING")
print("=" * 50)

# Simulating model evaluation accuracy metric
model_accuracy = 0.87

if model_accuracy >= 0.90:
    deployment_tier = "Tier-1: Production Auto-Deploy"
elif model_accuracy >= 0.80:
    deployment_tier = "Tier-2: Staging Review Required"
elif model_accuracy >= 0.60:
    deployment_tier = "Tier-3: Retraining Needed"
else:
    deployment_tier = "Tier-4: Pipeline Rejection"

print(f"Accuracy: {model_accuracy * 100:.1f}% -> Status: {deployment_tier}")

print("\n" + "=" * 50)
print("2. TERNARY OPERATOR (CONDITIONAL EXPRESSION)")
print("=" * 50)

gpu_available = True
# syntax: <value_if_true> if <condition> else <value_if_false>
device = "cuda:0" if gpu_available else "cpu"
print(f"Target compute hardware selected: {device}")

print("\n" + "=" * 50)
print("3. LOGICAL OPERATORS & SHORT-CIRCUITING")
print("=" * 50)

# In 'and', execution stops at first False; in 'or', stops at first True
dataset_loaded = True
batch_size = 64

if dataset_loaded and batch_size > 0:
    print("Pre-checks passed: Training worker initialized.")
else:
    print("Invalid batch setup or missing dataset.")

print("\n" + "=" * 50)
print("4. STRUCTURAL PATTERN MATCHING (MATCH-CASE)")
print("=" * 50)

# Modern, clean alternative to nested if-else ladders
http_status = 404

match http_status:
    case 200:
        action = "Data stream fetched successfully."
    case 400 | 422:
        action = "Malformed request payload."
    case 404:
        action = "Model endpoint / resource not found."
    case 500:
        action = "Internal inference engine error."
    case _:
        action = f"Unhandled status code: {http_status}"

print(f"API Code {http_status}: {action}")

print("\nExecution complete: Conditional logic flows verified.")