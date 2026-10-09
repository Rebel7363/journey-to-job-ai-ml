import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer implemented from mathematical first principles.
    Forward pass computes:
        h = W_0 * x + (alpha / r) * (B * A * x)
    where W_0 is frozen, A is initialized with Gaussian noise, and B is zero-initialized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        lora_alpha: float = 8.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # 1. Base linear layer (Frozen weights)
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # 2. Low-rank decomposition matrices (Trainable adapters)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize Matrix A with Kaiming uniform and B to 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        adapter_in = self.lora_dropout(x)
        adapter_out = (adapter_in @ self.lora_A.t()) @ self.lora_B.t()
        adapter_out = adapter_out * self.scaling
        return base_out + adapter_out


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Running LoRA Layer Scratch Verification on {device} ---\n")

    in_dim = 128
    out_dim = 128
    rank = 4
    alpha = 8.0

    lora_layer = LoRALinear(
        in_features=in_dim,
        out_features=out_dim,
        rank=rank,
        lora_alpha=alpha,
    ).to(device)

    # Calculate parameter statistics
    total_params = sum(p.numel() for p in lora_layer.parameters())
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Base Dimensions:          {in_dim} -> {out_dim}")
    print(f"LoRA Rank (r):            {rank} | Alpha: {alpha}")
    print(f"Total Parameters:         {total_params:,}")
    print(f"Frozen Parameters:        {frozen_params:,} ({(frozen_params / total_params) * 100:.1f}%)")
    print(f"Trainable Parameters:     {trainable_params:,} ({(trainable_params / total_params) * 100:.1f}%)")

    # Verification 1: Zero-init output equivalence
    dummy_input = torch.randn(2, 10, in_dim, device=device)
    with torch.no_grad():
        base_only = lora_layer.base_layer(dummy_input)
        lora_init_out = lora_layer(dummy_input)

    assert torch.allclose(base_only, lora_init_out, atol=1e-6), "Initial LoRA output deviates from base weights!"
    print("\nVerification Passed 1: B=0 initialization ensures exact base output preservation at step 0.")

    # Verification 2: Gradient Flow Check (computed with active autograd)
    train_input = torch.randn(2, 10, in_dim, device=device)
    train_out = lora_layer(train_input)
    loss = train_out.sum()
    loss.backward()

    assert lora_layer.base_layer.weight.grad is None, "Base layer received unexpected gradients!"
    assert lora_layer.lora_A.grad is not None, "LoRA matrix A did not receive gradients!"
    assert lora_layer.lora_B.grad is not None, "LoRA matrix B did not receive gradients!"
    print("Verification Passed 2: Gradients flow strictly into low-rank matrices A and B while base weights remain untouched.")


if __name__ == "__main__":
    main()