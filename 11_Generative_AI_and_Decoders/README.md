# Module 11: Generative AI, Decoder Architectures & Pre-trained Transformers

This module covers the principles of modern Generative AI and Large Language Model (LLM) foundations: moving from lower-triangular causal masking and Pre-LN decoder blocks to training an autoregressive character-level Mini-GPT from scratch, followed by production ecosystem integration with Hugging Face.

---

## Architecture & Code Progression

| Script | Domain | Core Implementation Mechanics |
| :--- | :--- | :--- |
| `01_causal_decoder_block.py` | Causal Attention & Decoders | Masked Multi-Head Attention, non-trainable lower-triangular causal buffer registration (`register_buffer`), Pre-LN Transformer decoder layers, greedy step-by-step next-token rollout. |
| `02_mini_gpt_language_model.py` | End-to-End LLM Architecture | Character-level tokenizer, block-shifting dataset loader (`x` vs `y = x[1:]`), learned positional embeddings, stacked GPT decoder blocks, cross-entropy loss convergence, temperature-scaled multinomial text generation. |
| `03_hf_transformers_pipeline.py` | Pre-trained Ecosystem | Tokenization workflows (`AutoTokenizer`), vocabulary dimension projection, causal logits extraction (`AutoModelForCausalLM`), and high-level pipeline inference using DistilGPT-2. |

---

## Theoretical & Mathematical Foundations

### 1. Autoregressive Probability Factorization
A decoder-only language model factorizes the joint probability of a sequence of tokens $X = (x_1, x_2, \dots, x_T)$ into a product of conditional probabilities using the probability chain rule:

$$P(X) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

At each forward pass, the model predicts the probability distribution of the next token $x_t$ given only the preceding context.

### 2. Causal Masking Dynamics
To prevent information leakage from future tokens during parallel training, self-attention scores are masked using a lower-triangular matrix $M$:

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where the mask $M_{i, j}$ is defined as:

$$M_{i, j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

Positions with $-\infty$ evaluate to zero attention weight after the softmax operation ($\lim_{z \to -\infty} e^z = 0$), strictly preserving temporal causality.

### 3. Sampling & Temperature Scaling
During generation, logits $z_i$ at the final sequence position are scaled by a temperature hyperparameter $T$ prior to probability normalization:

$$P(x_{\text{next}} = i) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

- **$T \to 0$ (Greedy / Argmax):** Concentrates probability mass on the single highest-scoring token, yielding deterministic responses.
- **Higher $T$:** Flattens the distribution, increasing generation diversity and variability.

---

## Verification & Execution

Execute the scripts from the repository root:

```bash
# 1. Causal Decoder Block & Greedy Generation Verification
python 11_Generative_AI_and_Decoders/01_causal_decoder_block.py

# 2. Train Mini-GPT & Run Sampling Loop
python 11_Generative_AI_and_Decoders/02_mini_gpt_language_model.py

# 3. Hugging Face Pre-trained Inference & Generation Pipeline
python 11_Generative_AI_and_Decoders/03_hf_transformers_pipeline.py