import os
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import torch


class UniformQuantizer:
    """
    Implements uniform symmetric and asymmetric quantization/dequantization
    from mathematical first principles.
    """

    @staticmethod
    def asymmetric_quantize(
        x: torch.Tensor, bits: int = 8
    ) -> tuple[torch.Tensor, float, int]:
        q_min = 0
        q_max = (1 << bits) - 1

        alpha = x.min().item()
        beta = x.max().item()

        # Prevent division by zero for uniform tensors
        scale = (beta - alpha) / (q_max - q_min) if beta != alpha else 1.0
        zero_point = round(-alpha / scale) + q_min
        zero_point = max(q_min, min(q_max, zero_point))

        # Quantize and clamp
        x_q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max).to(torch.uint8)
        return x_q, scale, zero_point

    @staticmethod
    def asymmetric_dequantize(
        x_q: torch.Tensor, scale: float, zero_point: int
    ) -> torch.Tensor:
        return scale * (x_q.float() - zero_point)

    @staticmethod
    def symmetric_quantize(
        x: torch.Tensor, bits: int = 8
    ) -> tuple[torch.Tensor, float]:
        q_max = (1 << (bits - 1)) - 1
        q_min = -q_max

        max_val = x.abs().max().item()
        scale = max_val / q_max if max_val != 0.0 else 1.0

        x_q = torch.clamp(torch.round(x / scale), q_min, q_max).to(torch.int8)
        return x_q, scale

    @staticmethod
    def symmetric_dequantize(x_q: torch.Tensor, scale: float) -> torch.Tensor:
        return scale * x_q.float()


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Uniform Quantization Verification on {device} ===\n")

    torch.manual_seed(42)
    # Simulate floating point weight distribution: N(0.5, 1.2^2)
    weights = torch.randn(512, 512, device=device) * 1.2 + 0.5

    # 1. Test Asymmetric INT8 Quantization
    q_asym, scale_asym, zp_asym = UniformQuantizer.asymmetric_quantize(weights, bits=8)
    deq_asym = UniformQuantizer.asymmetric_dequantize(q_asym, scale_asym, zp_asym)
    mse_asym = torch.mean((weights - deq_asym) ** 2).item()

    print("--- Asymmetric INT8 Quantization ---")
    print(f"Scale (S):               {scale_asym:.6f}")
    print(f"Zero-Point (Z):          {zp_asym}")
    print(f"Reconstruction MSE:      {mse_asym:.6e}")

    # 2. Test Symmetric INT8 Quantization
    q_sym, scale_sym = UniformQuantizer.symmetric_quantize(weights, bits=8)
    deq_sym = UniformQuantizer.symmetric_dequantize(q_sym, scale_sym)
    mse_sym = torch.mean((weights - deq_sym) ** 2).item()

    print("\n--- Symmetric INT8 Quantization ---")
    print(f"Scale (S):               {scale_sym:.6f}")
    print(f"Zero-Point (Z):          0 (Fixed)")
    print(f"Reconstruction MSE:      {mse_sym:.6e}")

    # 3. Compression footprint verification
    fp32_bytes = weights.element_size() * weights.nelement()
    int8_bytes = q_asym.element_size() * q_asym.nelement()
    compression_ratio = fp32_bytes / int8_bytes

    print("\n--- Memory Compression Footprint ---")
    print(f"FP32 Tensor Memory:      {fp32_bytes / 1024:.2f} KB")
    print(f"INT8 Tensor Memory:      {int8_bytes / 1024:.2f} KB")
    print(f"Compression Ratio:       {compression_ratio:.2f}x reduction")

    assert compression_ratio == 4.0, "Expected exact 4x compression from FP32 to INT8!"
    assert mse_asym < 1e-3, f"Asymmetric quantization MSE too high: {mse_asym}"
    assert mse_sym < 1e-3, f"Symmetric quantization MSE too high: {mse_sym}"

    print("\nVerification Passed: Uniform quantization faithfully preserved weights with 4x memory compression.")


if __name__ == "__main__":
    main()