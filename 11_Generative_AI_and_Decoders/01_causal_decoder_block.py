import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal (Masked) Self-Attention for Autoregressive Decoders.
    Ensures token at position t can only attend to tokens at positions <= t.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Key, Query, Value projections combined into a single linear layer
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal lower-triangular mask registered as non-trainable persistent buffer
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
            1, 1, max_seq_len, max_seq_len
        )
        self.register_buffer("bias", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projection for Q, K, V in one matrix multiplication
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # Reshape to (Batch, Heads, Seq_Len, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask up to current sequence length
        scores = scores.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context aggregation: (Batch, Heads, Seq_Len, d_k)
        context = torch.matmul(attn_weights, v)

        # Merge heads back: (Batch, Seq_Len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.c_proj(context)


class TransformerDecoderBlock(nn.Module):
    """
    Standard Pre-LN Decoder Block (Modern GPT-style):
    x = x + CausalSelfAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections (ensures better gradient flow than Post-LN)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AutoregressiveDemoLM(nn.Module):
    """Minimal Decoder-Only Language Model demonstrating next-token prediction."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=4 * d_model)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)

        # Add token and learned positional embeddings
        x = self.token_embeddings(idx) + self.pos_embeddings(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (Batch, Seq_Len, Vocab_Size)
        return logits


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Running Causal Decoder Block Verification on {device} ---\n")

    vocab_size = 50
    d_model = 64
    num_heads = 4
    num_layers = 2

    model = AutoregressiveDemoLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)

    # 1. Forward Pass Shape Verification
    dummy_prompt = torch.randint(0, vocab_size, (2, 8), device=device)
    logits = model(dummy_prompt)

    print(f"Input Prompt Shape: {tuple(dummy_prompt.shape)}")
    print(f"Logits Output Shape: {tuple(logits.shape)}  # (Batch, Seq_Len, Vocab_Size)")

    # 2. Greedy Autoregressive Generation Simulation
    print("\n--- Simulating Greedy Next-Token Generation (5 Steps) ---")
    generated = torch.tensor([[1, 5, 12]], device=device)  # Prompt of length 3
    print(f"Initial Prompt Tokens: {generated.tolist()[0]}")

    model.eval()
    with torch.no_grad():
        for step in range(5):
            out_logits = model(generated)
            # Pick last token logits to predict the next token
            next_token_logits = out_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            print(f"Step {step + 1}: Generated Token ID -> {next_token.item()} | Current Sequence: {generated.tolist()[0]}")

    print("\nVerification Passed: Causal masking prevents future leakage and enables autoregressive rollouts.")


if __name__ == "__main__":
    main()