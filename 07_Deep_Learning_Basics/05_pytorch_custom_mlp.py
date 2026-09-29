import torch
import torch.nn as nn
import torch.optim as optim


class MultiLayerPerceptron(nn.Module):
    """Deep Neural Network architecture with customizable hidden dimensions."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def generate_classification_data(
    num_samples: int = 300, in_features: int = 4, num_classes: int = 3, device="cpu"
):
    torch.manual_seed(42)
    # Synthetic feature matrix
    X = torch.randn(num_samples, in_features, device=device)
    # Synthetic ground-truth class indices
    y = torch.randint(0, num_classes, (num_samples,), device=device)
    return X, y


def train_and_evaluate(device: torch.device):
    in_features = 4
    hidden_dim = 16
    num_classes = 3
    epochs = 120
    learning_rate = 0.05

    X, y = generate_classification_data(device=device)

    model = MultiLayerPerceptron(in_features, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"--- Training Custom MLP on Device: {device} ---")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0 or epoch == epochs:
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y).float().mean().item() * 100
            print(
                f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | Training Accuracy: {accuracy:.2f}%"
            )

    # Model inference evaluation
    model.eval()
    with torch.no_grad():
        test_sample = torch.randn(1, in_features, device=device)
        raw_output = model(test_sample)
        probabilities = torch.softmax(raw_output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print("\n--- Single Sample Inference ---")
    print(f"Class Probabilities: {probabilities.cpu().numpy().round(4)}")
    print(f"Predicted Class Index: {predicted_class}")


if __name__ == "__main__":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    train_and_evaluate(device)