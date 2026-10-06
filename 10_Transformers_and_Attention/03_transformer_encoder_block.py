import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size = q.size(0)

        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    """Standard 2-layer MLP expanding to d_ff and projecting back to d_model."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer:
    Input -> Multi-Head Self-Attention -> Dropout -> Add & LayerNorm
          -> FeedForward Network       -> Dropout -> Add & LayerNorm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Sub-layer 1: Self-Attention with residual connection & pre/post LayerNorm
        attn_out = self.self_attn(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: FeedForward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Running Transformer Encoder Layer Verification on {device} ---\n")

    batch_size = 4
    seq_len = 8
    d_model = 64
    num_heads = 4
    d_ff = 256

    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    ).to(device)

    # Simulated input embeddings with positional encodings added
    dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
    output = encoder_layer(dummy_input)

    print(f"Input Tensor Shape:          {tuple(dummy_input.shape)}")
    print(f"Encoder Layer Output Shape:  {tuple(output.shape)}")

    assert output.shape == dummy_input.shape, "Shape mismatch after passing through encoder layer!"
    print("\nVerification Passed: Output shape perfectly preserved across LayerNorm and FeedForward sub-layers.")


if __name__ == "__main__":
    main()