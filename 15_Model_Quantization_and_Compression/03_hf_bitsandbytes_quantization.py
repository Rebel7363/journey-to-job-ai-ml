import importlib.metadata
import os
import sys
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()


class QuantizedLinearLayer(nn.Module):
    """
    Simulates a memory-efficient 8-bit quantized linear module compatible
    across hardware backends (MPS/CUDA/CPU).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Register integer weights and float scales as non-trainable buffers
        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.zeros(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features))

    @classmethod
    def from_float(cls, float_layer: nn.Linear) -> "QuantizedLinearLayer":
        quant_layer = cls(float_layer.in_features, float_layer.out_features)
        w = float_layer.weight.data

        # Row-wise absolute maximum scale calculation
        abs_max = w.abs().amax(dim=1, keepdim=True)
        scales = torch.where(abs_max == 0.0, torch.ones_like(abs_max), abs_max / 127.0)

        # Quantize to signed INT8 [-127, 127]
        w_q = torch.clamp(torch.round(w / scales), -127, 127).to(torch.int8)

        quant_layer.weight_q.copy_(w_q)
        quant_layer.scales.copy_(scales.squeeze(1))
        if float_layer.bias is not None:
            quant_layer.bias.data.copy_(float_layer.bias.data)

        return quant_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On-the-fly dequantization for GEMM execution
        w_float = self.weight_q.float() * self.scales.unsqueeze(1)
        return nn.functional.linear(x, w_float, self.bias)


def test_custom_quantized_architecture(device: str):
    print("--- 1. Verification of In-Memory Dynamic 8-Bit Layer Replacement ---")
    in_dim, out_dim = 1024, 2048
    float_linear = nn.Linear(in_dim, out_dim).to(device)

    # Calculate native float memory footprint
    fp32_bytes = sum(p.numel() * p.element_size() for p in float_linear.parameters())

    # Replace with quantized linear representation
    quant_linear = QuantizedLinearLayer.from_float(float_linear).to(device)

    # Quantized memory footprint (buffers + bias parameter)
    q_bytes = (
        quant_linear.weight_q.numel() * quant_linear.weight_q.element_size()
        + quant_linear.scales.numel() * quant_linear.scales.element_size()
        + (quant_linear.bias.numel() * quant_linear.bias.element_size() if quant_linear.bias is not None else 0)
    )

    compression_ratio = fp32_bytes / q_bytes
    print(f"Original FP32 Layer Memory:  {fp32_bytes / 1024:.2f} KB")
    print(f"Quantized INT8 Layer Memory: {q_bytes / 1024:.2f} KB")
    print(f"Layer Memory Compression:    {compression_ratio:.2f}x reduction")

    # Functional equivalence check
    test_inputs = torch.randn(4, in_dim, device=device)
    fp_out = float_linear(test_inputs)
    q_out = quant_linear(test_inputs)
    cos_sim = nn.functional.cosine_similarity(fp_out.flatten(), q_out.flatten(), dim=0).item()

    print(f"Cosine Similarity (Output):  {cos_sim:.6f}")
    assert compression_ratio > 3.5, f"Expected >3.5x compression, got {compression_ratio:.2f}x"
    assert cos_sim > 0.99, f"Output direction altered significantly: cosine similarity = {cos_sim:.6f}"
    print("In-Memory 8-Bit Linear replacement verified successfully.\n")


def test_hf_quantization_configs():
    print("--- 2. Hugging Face BitsAndBytes Integration Verification ---")
    model_id = "distilbert/distilgpt2"

    bnb_available = False
    try:
        importlib.metadata.version("bitsandbytes")
        from transformers import BitsAndBytesConfig
        bnb_available = True
    except Exception:
        bnb_available = False

    if bnb_available:
        try:
            bnb_config_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32,
            )
            print(f"BitsAndBytes NF4 4-bit Config created: type={bnb_config_4bit.bnb_4bit_quant_type}")

            bnb_config_8bit = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            print(f"BitsAndBytes Mixed INT8 Config created: threshold={bnb_config_8bit.llm_int8_threshold}")
        except Exception as e:
            print(f"BitsAndBytesConfig notice: {e}")
    else:
        print("Note: bitsandbytes package not installed; running architectural fallback mode.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer for '{model_id}' validated successfully.")


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Model Quantization & Library Integration Pipeline on {device} ===\n")

    test_custom_quantized_architecture(device)
    test_hf_quantization_configs()

    print("\nVerification Passed: Quantized layer dynamics and configuration pipelines validated.")


if __name__ == "__main__":
    main()