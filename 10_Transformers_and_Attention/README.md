# Module 10: Transformers & Multi-Head Self-Attention Architecture

This module implements the foundational components of the Transformer architecture (Vaswani et al., *"Attention Is All You Need"*) from mathematical first principles in PyTorch.

---

## Architectural Progression

| Script | Component | Key Technical Mechanics |
| :--- | :--- | :--- |
| `01_positional_encoding.py` | Sinusoidal Positional Encoding | Deterministic sinusoidal/cosinusoidal encodings injected into embeddings, log-space geometric progression, non-trainable persistent buffer registration (`register_buffer`). |
| `02_multi_head_attention.py` | Multi-Head Attention (MHA) | Linear projections ($W_Q, W_K, W_V$), multi-head splitting along hidden dimensions ($d_k = d_{\text{model}} / h$), parallel dot-product scaling, causal/padding mask support, output projection $W_O$. |
| `03_transformer_encoder_block.py` | Transformer Encoder Layer | Multi-Head Self-Attention sub-layer, residual skip connections, Pre/Post Layer Normalization (`LayerNorm`), and Position-wise FeedForward Network (GELU MLP expansion $d_{\text{model}} \to d_{ff} \to d_{\text{model}}$). |

---

## Mathematical Foundations

### 1. Sinusoidal Positional Encoding
Because Transformers process sequence tokens in parallel without recurrent step loops, spatial ordering must be injected:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where $pos$ is the token sequence index, $i$ is the channel index, and $d_{\text{model}}$ is the hidden state dimensionality.

### 2. Multi-Head Attention Mechanism
Instead of performing a single attention pooling over $d_{\text{model}}$, queries, keys, and values are linearly projected $h$ times to lower-dimensional sub-spaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i$$

### 3. Encoder Layer Flow & Residual Normalization
Each Encoder Block maintains tensor dimensionality ($B, S, d_{\text{model}}$) through residual addition and layer normalization:

$$\text{SubLayer}_1(x) = \text{LayerNorm}(x + \text{Dropout}(\text{MultiHeadSelfAttention}(x)))$$

$$\text{SubLayer}_2(x) = \text{LayerNorm}(\text{SubLayer}_1(x) + \text{Dropout}(\text{FFN}(\text{SubLayer}_1(x))))$$

$$\text{where} \quad \text{FFN}(z) = \max(0, z W_1 + b_1) W_2 + b_2 \quad \text{or} \quad \text{GELU}(z W_1 + b_1) W_2 + b_2$$

---

## Verification & Execution

To test the transformer building blocks on Apple Metal (`mps`) or CPU:

```bash
# 1. Verify Positional Encodings
python 10_Transformers_and_Attention/01_positional_encoding.py

# 2. Verify Multi-Head Attention Projection
python 10_Transformers_and_Attention/02_multi_head_attention.py

# 3. Verify Complete Transformer Encoder Block
python 10_Transformers_and_Attention/03_transformer_encoder_block.py