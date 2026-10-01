import os
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNet(nn.Module):
    """Feedforward network architecture for checkpointing demonstration."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 12, output_dim: int = 2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        return self.layer2(x)


def save_checkpoint(filepath: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint successfully saved to: {filepath}")


def load_checkpoint(filepath: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Checkpoint restored from: {filepath} (Trained up to Epoch {epoch} with Loss: {loss:.4f})")
    return epoch, loss


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- PyTorch Model Checkpointing Demo on {device} ---")

    checkpoint_path = "model_checkpoint.pth"

    # 1. Setup Architecture, Optimizer & Loss
    model = NeuralNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Synthetic Dummy Data
    X_train = torch.randn(50, 5, device=device)
    y_train = torch.randn(50, 2, device=device)

    # 2. Simulate Training for 15 epochs
    last_loss = 0.0
    for epoch in range(1, 16):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

    print(f"Training completed for 15 epochs. Final Loss: {last_loss:.4f}")

    # 3. Save State Checkpoint
    save_checkpoint(checkpoint_path, model, optimizer, epoch=15, loss=last_loss)

    # 4. Generate prediction before wiping model
    sample_input = torch.randn(1, 5, device=device)
    model.eval()
    with torch.no_grad():
        original_output = model(sample_input)

    # 5. Instantiate Fresh Model & Load Checkpoint
    fresh_model = NeuralNet().to(device)
    fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=0.01)

    load_checkpoint(checkpoint_path, fresh_model, fresh_optimizer, device)

    # 6. Verify Exact Output Match
    fresh_model.eval()
    with torch.no_grad():
        restored_output = fresh_model(sample_input)

    assert torch.allclose(original_output, restored_output, atol=1e-6), "Outputs do not match!"
    print("\nVerification Passed: Restored model produces identical inference outputs!")

    # Clean up generated file so repo stays neat
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Cleaned up temporary checkpoint file: {checkpoint_path}")


if __name__ == "__main__":
    main()