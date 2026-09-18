"""
Day 3: Functions, Scope & Advanced Signatures
---------------------------------------------
Demonstrating:
1. Type hints and default arguments
2. Dynamic parameter handling (*args and **kwargs)
3. Return unpacking and multi-value outputs
4. Anonymous functions (lambda)
5. Variable scope mechanics (Local vs Global)
"""

print("=" * 50)
print("1. TYPE HINTS & DEFAULT ARGUMENTS")
print("=" * 50)

def calculate_loss(actual: float, predicted: float, penalty: float = 0.0) -> float:
    """Computes basic absolute error with an optional regularization penalty."""
    error = abs(actual - predicted) + penalty
    return round(error, 4)

base_error = calculate_loss(1.0, 0.82)
penalized_error = calculate_loss(1.0, 0.82, penalty=0.05)

print(f"Base Loss:      {base_error}")
print(f"Penalized Loss: {penalized_error}")

print("\n" + "=" * 50)
print("2. DYNAMIC ARGUMENTS (*args and **kwargs)")
print("=" * 50)

def train_model(model_name: str, *metrics: str, **hyperparameters) -> None:
    """Accepts arbitrary positional metrics and keyword hyperparameters."""
    print(f"Model Architecture: {model_name}")
    print(f"Tracking Metrics:   {list(metrics)}")
    print("Hyperparameters:")
    for key, val in hyperparameters.items():
        print(f"  - {key}: {val}")

# Passing variable metrics (*args) and keyword configs (**kwargs)
train_model(
    "ResNet-50",
    "accuracy", "f1_score", "val_loss",
    learning_rate=0.001,
    batch_size=32,
    optimizer="AdamW"
)

print("\n" + "=" * 50)
print("3. MULTI-VALUE RETURNS & UNPACKING")
print("=" * 50)

def evaluate_distribution(values: list[float]) -> tuple[float, float]:
    """Calculates min and max boundary points."""
    return min(values), max(values)

bounds = [0.12, 0.89, 0.45, 0.99, 0.03]
min_val, max_val = evaluate_distribution(bounds)
print(f"Extracted Range: [{min_val} to {max_val}]")

print("\n" + "=" * 50)
print("4. ANONYMOUS LAMBDA EXPRESSIONS")
print("=" * 50)

# Quick utility transformations (common inside data pipelines)
normalize = lambda x, max_x: round(x / max_x, 3)
raw_signal = 180
print(f"Normalized Signal (180/255): {normalize(raw_signal, 255)}")

print("\n" + "=" * 50)
print("5. VARIABLE SCOPE (LOCAL VS GLOBAL)")
print("=" * 50)

global_pipeline_state = "IDLE"

def run_job() -> str:
    local_state = "RUNNING"
    return f"Inside Function -> Local: {local_state} | Global: {global_pipeline_state}"

print(run_job())
print(f"Outside Function -> Global remains: {global_pipeline_state}")

print("\nExecution complete: Function mechanics and scope validated.")