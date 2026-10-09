import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Scratch LoRA Linear Layer
class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        lora_alpha: float = 8.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = lora_alpha / rank

        # Freeze existing base layer weights
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Adapter low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        adapter_in = self.lora_dropout(x)
        adapter_out = (adapter_in @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + adapter_out


# 2. Simulated Pretrained Attention Block
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.out_proj(context)


# 3. Dynamic Adapter Injection Helper
def inject_lora_into_model(model: nn.Module, target_layers=("q_proj", "v_proj"), rank=4, alpha=8.0):
    """
    Traverses the model hierarchy, freezes parameters, and wraps targeted
    linear projection layers with low-rank adapters.
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_children():
        if name in target_layers and isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, rank=rank, lora_alpha=alpha))
        else:
            inject_lora_into_model(module, target_layers=target_layers, rank=rank, alpha=alpha)


def get_parameter_summary(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Simulating LoRA Parameter-Efficient Fine-Tuning on {device} ---\n")

    d_model = 256
    model = MultiHeadSelfAttentionBlock(d_model=d_model).to(device)

    # Pre-injection baseline
    total_before, trainable_before, _ = get_parameter_summary(model)
    print(f"Before LoRA Injection:")
    print(f"  Total Params:     {total_before:,}")
    print(f"  Trainable Params: {trainable_before:,} (100.0%)\n")

    # Inject LoRA adapters to q_proj and v_proj
    inject_lora_into_model(model, target_layers=("q_proj", "v_proj"), rank=4, alpha=8.0)
    model.to(device)

    total_after, trainable_after, frozen_after = get_parameter_summary(model)
    reduction_pct = (1.0 - (trainable_after / total_before)) * 100.0

    print(f"After LoRA Injection (Targets: q_proj, v_proj | Rank: 4):")
    print(f"  Total Params:     {total_after:,}")
    print(f"  Frozen Base:      {frozen_after:,} ({(frozen_after / total_after) * 100:.2f}%)")
    print(f"  Trainable LoRA:   {trainable_after:,} ({(trainable_after / total_after) * 100:.2f}%)")
    print(f"  Parameter Reduction: {reduction_pct:.2f}% fewer trainable parameters vs Full Fine-Tuning!\n")

    # Simulate Fine-Tuning Optimization Step
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    dummy_input = torch.randn(4, 16, d_model, device=device)
    target = torch.randn(4, 16, d_model, device=device)

    model.train()
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    print(f"Fine-Tuning Simulation Step Loss: {loss.item():.4f}")
    assert model.k_proj.weight.grad is None, "Frozen projection received gradients!"
    assert model.q_proj.lora_A.grad is not None, "LoRA adapter failed to accumulate gradients!"
    print("\nVerification Passed: Only LoRA adapter parameters were updated; base weights remained frozen.")


if __name__ == "__main__":
    main()