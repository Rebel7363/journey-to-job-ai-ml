import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SyntheticVisionDataset(Dataset):
    """Synthetic dataset producing multi-class 3x32x32 image tensors with target signals."""

    def __init__(self, num_samples: int = 400, num_classes: int = 4, seed: int = 42):
        torch.manual_seed(seed)
        self.images = torch.randn(num_samples, 3, 32, 32)
        # Class assignment with clear channel bias for learnability
        channel_means = self.images.mean(dim=(2, 3))
        scores = channel_means[:, 0] * 2.0 - channel_means[:, 1] * 1.5 + channel_means[:, 2]
        self.labels = torch.bucketize(scores, boundaries=torch.tensor([-0.8, 0.0, 0.8]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class VisionClassifierCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_and_validate():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Training CNN Pipeline on {device} ---\n")

    # Data Partitioning
    dataset = SyntheticVisionDataset(num_samples=400, num_classes=4)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = VisionClassifierCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # Evaluation Loop
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for v_imgs, v_targets in val_loader:
                    v_imgs, v_targets = v_imgs.to(device), v_targets.to(device)
                    v_outputs = model(v_imgs)
                    v_loss = criterion(v_outputs, v_targets)

                    val_loss += v_loss.item() * v_imgs.size(0)
                    v_preds = torch.argmax(v_outputs, dim=1)
                    correct_val += (v_preds == v_targets).sum().item()
                    total_val += v_targets.size(0)

            epoch_val_loss = val_loss / total_val
            epoch_val_acc = (correct_val / total_val) * 100

            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.2f}%"
            )


if __name__ == "__main__":
    train_and_validate()