import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SyntheticClassificationDataset(Dataset):
    """Custom PyTorch Dataset with deterministic synthetic clusters."""

    def __init__(self, num_samples: int = 600, in_features: int = 6, num_classes: int = 3):
        torch.manual_seed(42)
        self.features = torch.randn(num_samples, in_features)
        
        # Create separable target classes based on feature combinations
        class_signals = (
            self.features[:, 0] * 1.5
            - self.features[:, 1] * 0.8
            + torch.randn(num_samples) * 0.2
        )
        self.labels = torch.bucketize(
            class_signals,
            boundaries=torch.tensor([-0.5, 0.5])
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_batch_training():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Executing Mini-Batch Training on Device: {device}\n")

    # 1. Dataset & DataLoader Setup
    dataset = SyntheticClassificationDataset(num_samples=600, in_features=6, num_classes=3)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = SimpleClassifier(in_features=6, hidden_dim=16, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. Training Loop with Mini-Batches
    epochs = 40
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_loss = total_train_loss / total_train
        epoch_acc = (correct_train / total_train) * 100

        if epoch % 10 == 0 or epoch == epochs:
            # Validation Evaluation
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    v_x, v_y = v_x.to(device), v_y.to(device)
                    v_outputs = model(v_x)
                    v_preds = torch.argmax(v_outputs, dim=1)
                    correct_val += (v_preds == v_y).sum().item()
                    total_val += v_y.size(0)

            val_acc = (correct_val / total_val) * 100
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:5.2f}% | Val Acc: {val_acc:5.2f}%")


if __name__ == "__main__":
    run_batch_training()