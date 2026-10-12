import os
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniRewardModel(nn.Module):
    """
    Reward model architecture: Maps sequence representations
    to a scalar utility score r_theta(x, y).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.reward_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 1. Embedding lookup
        x = self.embedding(input_ids)
        # 2. Representation encoding
        h = self.transformer_layer(x)
        # 3. Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # 4. Scalar score projection
        return self.reward_head(pooled).squeeze(-1)


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """
    Pairwise ranking cross-entropy loss derived from Bradley-Terry preference model:
    Loss = -log(sigmoid(r_chosen - r_rejected))
    """
    diff = r_chosen - r_rejected
    # log(sigmoid(x)) = -log(1 + exp(-x)) -> softplus(-x) for numerical stability
    return F.softplus(-diff).mean()


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Bradley-Terry Reward Modeling Verification on {device} ===\n")

    torch.manual_seed(42)
    model = MiniRewardModel(d_model=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Synthetic batch: Prompt paired with Chosen (y_w) vs Rejected (y_l)
    batch_size, seq_len = 8, 12
    chosen_ids = torch.randint(1, 999, (batch_size, seq_len), device=device)
    rejected_ids = torch.randint(1, 999, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, device=device)

    # Initial score evaluation
    with torch.no_grad():
        init_r_chosen = model(chosen_ids, mask)
        init_r_rejected = model(rejected_ids, mask)
        init_acc = (init_r_chosen > init_r_rejected).float().mean().item()

    print(f"Initial Mean Chosen Reward:   {init_r_chosen.mean().item():.4f}")
    print(f"Initial Mean Rejected Reward: {init_r_rejected.mean().item():.4f}")
    print(f"Initial Pairwise Accuracy:    {init_acc * 100:.1f}%\n")

    # Optimization steps on preference pairs
    model.train()
    for step in range(1, 21):
        optimizer.zero_grad()
        r_w = model(chosen_ids, mask)
        r_l = model(rejected_ids, mask)
        loss = bradley_terry_loss(r_w, r_l)
        loss.backward()
        optimizer.step()

    final_r_chosen = model(chosen_ids, mask)
    final_r_rejected = model(rejected_ids, mask)
    final_acc = (final_r_chosen > final_r_rejected).float().mean().item()

    print(f"Step 20 Pairwise Loss:        {loss.item():.4f}")
    print(f"Final Mean Chosen Reward:     {final_r_chosen.mean().item():.4f}")
    print(f"Final Mean Rejected Reward:   {final_r_rejected.mean().item():.4f}")
    print(f"Final Pairwise Accuracy:      {final_acc * 100:.1f}%")

    assert final_r_chosen.mean().item() > final_r_rejected.mean().item(), "Reward model failed to separate preferences!"
    assert final_acc >= 0.85, "Preference alignment accuracy below expected threshold!"
    print("\nVerification Passed: Bradley-Terry pairwise ranking loss correctly separated preferred completions.")


if __name__ == "__main__":
    main()