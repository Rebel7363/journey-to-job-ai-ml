# Module 15: Model Quantization & Compression

This module covers the core mathematical mechanics and production implementations of Post-Training Quantization (PTQ) and low-bit weight representation. It covers uniform affine/symmetric quantization from scratch, block-wise outlier-resistant AbsMax scaling, dynamic in-memory INT8 linear layer replacement, and Hugging Face `BitsAndBytesConfig` (NF4/FP4 and LLM.int8()) orchestration.

---

## Technical Implementations

| Script | Domain | Key Architectural Mechanics |
| :--- | :--- | :--- |
| `01_uniform_quantization_scratch.py` | Mathematical Foundations | Asymmetric (affine) and symmetric uniform quantization, dynamic scale factor $S$ and integer zero-point $Z$ derivation, clamping boundaries, and exact $4\times$ memory footprint reduction from FP32 to INT8. |
| `02_vector_absmax_and_block_quant.py` | Outlier Mitigation | Block-wise (chunk-based) AbsMax quantization, dynamic tile scale computation, emergent activation outlier isolation, and dot-product error accumulation analysis. |
| `03_hf_bitsandbytes_quantization.py` | Production & Library Integration | Custom dynamic `QuantizedLinearLayer` module with non-trainable INT8 buffer storage, on-the-fly dequantization GEMM execution, and Hugging Face `BitsAndBytesConfig` (NF4, Double Quant, LLM.int8() threshold) orchestration. |

---

## Mathematical Formulation

### 1. Uniform Affine (Asymmetric) Quantization
Given continuous floating-point values bounded by $[\alpha, \beta]$ and a target integer bit-width $b$ with discrete range $[q_{\min}, q_{\max}]$:

$$S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}, \quad Z = \operatorname{round}\left(-\frac{\alpha}{S}\right) + q_{\min}$$

- **Quantization:**
  $$x_q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{S}\right) + Z, q_{\min}, q_{\max}\right)$$

- **Dequantization:**
  $$\hat{x} = S \cdot (x_q - Z)$$

### 2. Symmetric Quantization
Locks the zero-point to zero ($Z = 0$), forcing the dynamic range to be symmetric around the origin:

$$S = \frac{\max(\vert{}x\vert{})}{q_{\max}}, \quad x_q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{S}\right), -q_{\max}, q_{\max}\right)$$

$$\hat{x} = S \cdot x_q$$

### 3. Block-wise AbsMax Scaling
Large language models exhibit emergent activation outliers that distort global scale factors. By partitioning a weight tensor $W \in \mathbb{R}^{M \times N}$ into contiguous blocks $\mathbf{b}_i \in \mathbb{R}^B$:

$$S_i = \frac{\max_{j \in [1, B]} \vert{}b_{i, j}\vert{}}{127}$$

$$b_{q, i, j} = \operatorname{clamp}\left(\operatorname{round}\left(\frac{b_{i, j}}{S_i}\right), -127, 127\right)$$

This confines the quantization noise of extreme outliers strictly within their respective blocks, yielding up to $>95\%$ lower mean squared reconstruction error compared to per-tensor scaling.

---

## Verification & Execution

Run all verification scripts from the repository root:

```bash
# 1. Verify uniform asymmetric/symmetric quantization & compression ratios
python 15_Model_Quantization_and_Compression/01_uniform_quantization_scratch.py

# 2. Verify block-wise AbsMax outlier resistance and linear forward pass
python 15_Model_Quantization_and_Compression/02_vector_absmax_and_block_quant.py

# 3. Verify in-memory INT8 linear layer replacement and Hugging Face config pipeline
python 15_Model_Quantization_and_Compression/03_hf_bitsandbytes_quantization.py