import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k) + Mask) @ V

    Shapes:
    q: (Batch, Heads, Seq_Len_Q, d_k)
    k: (Batch, Heads, Seq_Len_K, d_k)
    v: (Batch, Heads, Seq_Len_V, d_v) where Seq_Len_K == Seq_Len_V
    mask: Optional tensor broadcastable to (Batch, Heads, Seq_Len_Q, Seq_Len_K)
    """
    d_k = q.size(-1)

    # 1. Compute raw affinity scores: (Batch, Heads, Seq_Len_Q, Seq_Len_K)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply causal or padding mask if present
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3. Softmax across key sequence dimension
    attn_weights = F.softmax(scores, dim=-1)

    # 4. Context aggregation over values: (Batch, Heads, Seq_Len_Q, d_v)
    context = torch.matmul(attn_weights, v)

    return context, attn_weights


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Running Scaled Dot-Product Attention on {device} ---\n")

    batch_size = 2
    heads = 1
    seq_len = 4
    d_k = 8
    d_v = 8

    # Random Query, Key, Value representations
    torch.manual_seed(42)
    q = torch.randn(batch_size, heads, seq_len, d_k, device=device)
    k = torch.randn(batch_size, heads, seq_len, d_k, device=device)
    v = torch.randn(batch_size, heads, seq_len, d_v, device=device)

    # Unmasked Attention
    context_unmasked, weights_unmasked = scaled_dot_product_attention(q, k, v)
    print("--- 1. Bidirectional Attention (e.g., BERT-style) ---")
    print(f"Output Context Shape: {tuple(context_unmasked.shape)}")
    print(f"Sample Attention Weights Matrix (Batch 0, Head 0):\n{weights_unmasked[0, 0].cpu().numpy().round(3)}")

    # Causal / Autoregressive Masking (Lower Triangular Matrix)
    # Mask out future tokens so position i cannot attend to position j > i
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    context_masked, weights_masked = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    print("\n--- 2. Causal Masked Attention (e.g., GPT-style Autoregressive) ---")
    print(f"Causal Weights Matrix (Batch 0, Head 0):\n{weights_masked[0, 0].cpu().numpy().round(3)}")


if __name__ == "__main__":
    main()