import torch
import torch.nn as nn
import torch.optim as optim


class ConvNet(nn.Module):
    """
    Standard CNN architecture for image classification:
    [Conv2d -> ReLU -> MaxPool2d] x 2 -> Flatten -> Linear -> ReLU -> Linear
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Block 1: Input (B, 1, 28, 28) -> (B, 16, 14, 14)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: (B, 16, 14, 14) -> (B, 32, 7, 7)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier head: (B, 32 * 7 * 7) -> (B, 10)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


def inspect_feature_shapes(device: torch.device):
    # Dummy batch of 4 grayscale images of size 28x28
    dummy_input = torch.randn(4, 1, 28, 28, device=device)
    model = ConvNet(in_channels=1, num_classes=10).to(device)

    print(f"--- Input Shape: {tuple(dummy_input.shape)} ---")
    
    out_b1 = model.conv_block1(dummy_input)
    print(f"After Conv Block 1: {tuple(out_b1.shape)}")

    out_b2 = model.conv_block2(out_b1)
    print(f"After Conv Block 2: {tuple(out_b2.shape)}")

    logits = model(dummy_input)
    print(f"Final Model Logits Output: {tuple(logits.shape)}\n")


def simulate_training_step(device: torch.device):
    model = ConvNet(in_channels=1, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 16 dummy images (1 channel, 28x28) with random labels (0 to 9)
    x_batch = torch.randn(16, 1, 28, 28, device=device)
    y_batch = torch.randint(0, 10, (16,), device=device)

    print("--- Single Optimization Step Execution ---")
    model.train()
    optimizer.zero_grad()
    predictions = model(x_batch)
    loss = criterion(predictions, y_batch)
    loss.backward()
    optimizer.step()

    print(f"Batch Loss: {loss.item():.4f}")
    preds = torch.argmax(predictions, dim=1)
    print(f"Predicted Classes: {preds.tolist()}")
    print(f"Target Classes:    {y_batch.tolist()}")


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Running on Device: {device}\n")
    inspect_feature_shapes(device)
    simulate_training_step(device)