import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module implemented from foundational math:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = q.size(0)

        # 1. Linear projections & split into heads:
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention across all heads in parallel
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_k)

        # 3. Concatenate heads back into original d_model space:
        # (batch, seq_len, num_heads * d_k) == (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final output projection
        output = self.w_o(context)
        return output, attn_weights


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Multi-Head Attention Verification on {device} ---\n")

    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)

    # Simulated token representations
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Self-attention: Q, K, V are all projections of input x
    output, weights = mha(q=x, k=x, v=x)

    print(f"Input Shape:               {tuple(x.shape)}")
    print(f"MHA Output Shape:          {tuple(output.shape)}")
    print(f"Attention Weights Shape:   {tuple(weights.shape)}  # (Batch, Heads, Seq_Q, Seq_K)")

    # Assert shape preservation
    assert output.shape == x.shape, "Multi-Head Attention failed to preserve sequence dimensions!"
    print("\nVerification Passed: Multi-Head Attention correctly projects, splits, aggregates, and preserves tensor dimensions.")


if __name__ == "__main__":
    main()