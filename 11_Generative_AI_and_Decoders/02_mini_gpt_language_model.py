import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# 1. Dataset & Tokenizer Setup
CORPUS = """
PyTorch enables neural networks to compute gradients automatically.
Transformers use self-attention to model long-range sequence context.
Generative language models predict the next token autoregressively.
Deep learning architectures scale with parallel matrix operations.
"""


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str):
        return [self.char2idx[c] for c in s if c in self.char2idx]

    def decode(self, indices):
        return "".join([self.idx2char[i] for i in indices])


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        # Chunk of size block_size as input (x), chunk shifted by 1 as target (y)
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# 2. Causal Attention & GPT Block
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        causal_mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("bias", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# 3. Mini-GPT Architecture
class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, num_layers: int = 2, block_size: int = 32):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, num_heads, block_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 4. Training & Generation Demonstration
def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Training Mini-GPT Character Language Model on {device} ---\n")

    tokenizer = CharTokenizer(CORPUS)
    encoded_corpus = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)

    block_size = 24
    dataset = CharDataset(encoded_corpus, block_size=block_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        block_size=block_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-2)

    epochs = 40
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")

    # Generate text from a prompt
    prompt = "PyTorch"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    out_ids = model.generate(prompt_ids, max_new_tokens=60, temperature=0.7)
    generated_text = tokenizer.decode(out_ids[0].tolist())

    print("\n--- Autoregressive Text Generation Sample ---")
    print(f"Prompt:    '{prompt}'")
    print(f"Generated: {generated_text.strip()}")


if __name__ == "__main__":
    main()