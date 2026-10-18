import os
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F


def per_tensor_absmax(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric per-tensor AbsMax quantization to INT8."""
    max_val = x.abs().max().item()
    scale = max_val / 127.0 if max_val != 0.0 else 1.0
    x_q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return x_q, scale


def blockwise_absmax_quantize(
    x: torch.Tensor, block_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Divides flattened tensor into chunks of block_size and quantizes each block
    with its own independent scale factor.
    """
    orig_shape = x.shape
    flattened = x.flatten()
    num_elements = flattened.numel()

    # Calculate padding if not divisible by block_size
    pad_len = (block_size - (num_elements % block_size)) % block_size
    if pad_len > 0:
        flattened = F.pad(flattened, (0, pad_len), "constant", 0.0)

    # Reshape into [num_blocks, block_size]
    blocks = flattened.view(-1, block_size)

    # Compute scales per block
    max_per_block = blocks.abs().amax(dim=1, keepdim=True)
    scales = torch.where(
        max_per_block == 0.0,
        torch.ones_like(max_per_block),
        max_per_block / 127.0,
    )

    # Quantize blocks
    blocks_q = torch.clamp(torch.round(blocks / scales), -127, 127).to(torch.int8)

    # Remove padding and return to original shape
    quantized_flat = blocks_q.flatten()[:num_elements]
    quantized = quantized_flat.view(orig_shape)

    return quantized, scales.squeeze(1)


def blockwise_absmax_dequantize(
    x_q: torch.Tensor, scales: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """Reconstructs float tensor using block-level scales."""
    orig_shape = x_q.shape
    flattened_q = x_q.flatten()
    num_elements = flattened_q.numel()

    pad_len = (block_size - (num_elements % block_size)) % block_size
    if pad_len > 0:
        flattened_q = F.pad(flattened_q, (0, pad_len), "constant", 0)

    blocks_q = flattened_q.view(-1, block_size).float()
    dequant_blocks = blocks_q * scales.unsqueeze(1)

    dequant_flat = dequant_blocks.flatten()[:num_elements]
    return dequant_flat.view(orig_shape)


def quantized_linear_forward(
    x: torch.Tensor, w_q: torch.Tensor, w_scales: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """
    Simulates quantized linear forward pass:
    Dequantizes weights on-the-fly and executes high-precision GEMM.
    """
    w_dequant = blockwise_absmax_dequantize(w_q, w_scales, block_size=block_size)
    return torch.matmul(x, w_dequant.t())


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Block-wise AbsMax & Outlier Robustness on {device} ===\n")

    torch.manual_seed(42)
    # 1. Create a weight matrix with realistic LLM emergent activation outliers
    in_features, out_features = 256, 256
    weights = torch.randn(out_features, in_features, device=device) * 0.1

    # Inject extreme outliers in 0.5% of locations
    outlier_mask = torch.rand_like(weights) < 0.005
    weights[outlier_mask] = weights[outlier_mask] * 40.0

    print(f"Total weights:        {weights.numel()}")
    print(f"Injected outliers:    {outlier_mask.sum().item()}")
    print(f"Max absolute value:   {weights.abs().max().item():.4f}\n")

    # 2. Compare Per-Tensor vs Block-wise (Block size = 64)
    # Per-tensor
    w_per_tensor, scale_pt = per_tensor_absmax(weights)
    deq_per_tensor = w_per_tensor.float() * scale_pt
    mse_per_tensor = torch.mean((weights - deq_per_tensor) ** 2).item()

    # Block-wise
    block_sz = 64
    w_block, block_scales = blockwise_absmax_quantize(weights, block_size=block_sz)
    deq_block = blockwise_absmax_dequantize(w_block, block_scales, block_size=block_sz)
    mse_block = torch.mean((weights - deq_block) ** 2).item()

    print(f"Per-Tensor Quantization MSE:    {mse_per_tensor:.6e}")
    print(f"Block-wise Quantization MSE:    {mse_block:.6e}")
    improvement = (mse_per_tensor - mse_block) / mse_per_tensor * 100
    print(f"Block-wise Reconstruction Gain: {improvement:.2f}% MSE reduction")

    # 3. Simulate Linear Layer Forward Pass
    inputs = torch.randn(8, in_features, device=device)
    fp32_output = torch.matmul(inputs, weights.t())
    quant_output = quantized_linear_forward(inputs, w_block, block_scales, block_size=block_sz)

    output_error = torch.mean((fp32_output - quant_output) ** 2).item()
    print(f"\nLinear Forward Pass Output MSE: {output_error:.6e}")

    assert mse_block < mse_per_tensor, "Block-wise quantization failed to outperform per-tensor under outliers!"
    assert output_error < 1e-2, f"Quantized linear output deviation too high: {output_error}"
    print("\nVerification Passed: Block-wise quantization successfully mitigated outlier distortion.")


if __name__ == "__main__":
    main()