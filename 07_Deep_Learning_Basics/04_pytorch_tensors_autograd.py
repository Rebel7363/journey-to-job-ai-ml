import torch
import torch.nn as nn
import torch.optim as optim


def verify_environment():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Active Device: {device}\n")
    return device


def tensor_operations_demo(device):
    # Tensor initialization, shapes, and indexing
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    w = torch.randn(2, 2, requires_grad=True, device=device)

    # Autograd computation graph
    y = torch.matmul(x, w) + 2.0
    loss = y.sum()

    # Backpropagation
    loss.backward()

    print("--- Autograd Demo ---")
    print(f"Input X:\n{x}")
    print(f"Weights W:\n{w}")
    print(f"Loss Output: {loss.item():.4f}")
    print(f"Gradient dLoss/dW:\n{w.grad}\n")


def train_linear_model(device):
    # Synthetic dataset: y = 3.5 * X + 1.2
    torch.manual_seed(42)
    X = torch.randn(100, 1, device=device)
    y_true = 3.5 * X + 1.2 + 0.1 * torch.randn(100, 1, device=device)

    # Linear model: y = w * x + b
    model = nn.Linear(in_features=1, out_features=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    print("--- Training Linear Regression in PyTorch ---")
    for epoch in range(1, 101):
        # 1. Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y_true)

        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 3. Parameter update
        optimizer.step()

        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d}/100 | Loss: {loss.item():.6f}")

    [w_learned, b_learned] = model.parameters()
    print("\nTraining Complete:")
    print(f"Learned Weight: {w_learned.item():.4f} (True: 3.5000)")
    print(f"Learned Bias:   {b_learned.item():.4f} (True: 1.2000)")


if __name__ == "__main__":
    device = verify_environment()
    tensor_operations_demo(device)
    train_linear_model(device)