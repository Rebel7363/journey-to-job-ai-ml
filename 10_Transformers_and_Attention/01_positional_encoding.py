import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encodings:
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Division term in log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even indices -> Sine, Odd indices -> Cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as non-trainable buffer with batch dimension: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with position information added.
        """
        seq_len = x.size(1)
        # Add position embeddings up to seq_len
        return x + self.pe[:, :seq_len, :]


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Running Positional Encoding Verification on {device} ---")

    batch_size = 2
    seq_len = 5
    d_model = 16

    pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=100).to(device)

    # Simulated token embeddings
    dummy_embeddings = torch.zeros(batch_size, seq_len, d_model, device=device)
    encoded_embeddings = pos_encoder(dummy_embeddings)

    print(f"Input Embedding Shape:     {tuple(dummy_embeddings.shape)}")
    print(f"Position-Encoded Shape:    {tuple(encoded_embeddings.shape)}")

    # Check that position 0 has unique encoding vs position 1
    pos_0 = encoded_embeddings[0, 0, :].cpu().numpy().round(3)
    pos_1 = encoded_embeddings[0, 1, :].cpu().numpy().round(3)

    print(f"\nPosition 0 Vector (First 6 dims): {pos_0[:6]}")
    print(f"Position 1 Vector (First 6 dims): {pos_1[:6]}")

    # Validate deterministic consistency across batch
    assert torch.allclose(encoded_embeddings[0], encoded_embeddings[1]), "Batch elements differ!"
    print("\nVerification Passed: Positional encodings successfully generated and broadcasted across batch.")


if __name__ == "__main__":
    main()