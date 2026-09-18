"""
Day 3: Loops & Iteration Patterns
---------------------------------
Demonstrating:
1. Standard 'for' and 'while' loops
2. Enumerate pattern (index tracking without manual counters)
3. Parallel iteration using 'zip'
4. Loop control: break, continue, and loop 'else' clauses
5. Idiomatic List Comprehensions (data processing pattern)
"""

print("=" * 50)
print("1. ENUMERATE: TRACKING INDEX & VALUES")
print("=" * 50)

hyperparams = ["learning_rate", "batch_size", "dropout_rate", "optimizer"]

# Enumerate avoids manual index increment variables
for index, param in enumerate(hyperparams, start=1):
    print(f"Step {index}: Optimizing {param}")

print("\n" + "=" * 50)
print("2. ZIP: PARALLEL ITERATION")
print("=" * 50)

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
scores = [0.94, 0.91, 0.89, 0.90]

# Zip pairs corresponding elements across iterables

for metric, score in zip(metrics, scores):
    print(f"{metric:<12}: {score:.2f}")

print("\n" + "=" * 50)
print("3. LOOP CONTROL: BREAK, CONTINUE, & LOOP-ELSE")
print("=" * 50)

# Continue skips remaining body; break terminates immediately

loss_values = [0.85, 0.62, 0.45, 0.02, 0.38]
target_threshold = 0.05

for epoch, loss in enumerate(loss_values, start=1):
    if loss > 0.70:
        continue  # Skip logging unstable early loss
    
    print(f"Epoch {epoch}: Evaluating loss = {loss:.2f}")
    
    if loss <= target_threshold:
        print(f"Convergence target reached at Epoch {epoch}. Halting training early.")
        break
else:
    # Runs only if loop terminates naturally without hitting 'break'
    print("Training finished without early convergence trigger.")

print("\n" + "=" * 50)
print("4. WHILE LOOPS & LIST COMPREHENSIONS")
print("=" * 50)

# While loop with convergence condition

epoch_count = 1
max_epochs = 3
while epoch_count <= max_epochs:
    print(f"Running epoch {epoch_count}/{max_epochs}...")
    epoch_count += 1

# List comprehension: Pythonic transformation pattern

raw_durations = [120, 240, 360, 480]  # in seconds
minutes = [sec / 60 for sec in raw_durations]
print(f"Raw Seconds:  {raw_durations}")
print(f"Into Minutes: {minutes}")

print("\nExecution complete: Iteration mechanisms validated.")