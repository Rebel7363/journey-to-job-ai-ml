# Module 12: Parameter-Efficient Fine-Tuning (PEFT) & LoRA

This module implements **Low-Rank Adaptation (LoRA)** (Hu et al., 2021) from mathematical first principles in PyTorch and integrates it with the industry-standard Hugging Face PEFT ecosystem. It demonstrates how parameter-efficient fine-tuning drastically reduces trainable parameter overhead during LLM adaptation while preserving full model representation capacity.

---

## Technical Implementations

| Script | Domain | Key Architectural Mechanics |
| :--- | :--- | :--- |
| `01_lora_linear_scratch.py` | LoRA Layer Math from Scratch | Frozen base projection $W_0$, low-rank decomposition matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$, scaling hyperparameter $\frac{\alpha}{r}$, $B=0$ initialization proof, autograd gradient isolation check. |
| `02_peft_lora_finetune.py` | Dynamic Injection & Fine-Tuning | Recursive model graph traversal, selective projection replacement (`q_proj`, `v_proj`), 90%+ trainable parameter reduction calculation, and end-to-end backpropagation loop verifying frozen base invariance. |
| `03_hf_peft_lora_finetune.py` | Production Ecosystem Integration | Hugging Face `peft` (`LoraConfig`, `get_peft_model`), targeting attention projection weights (`c_attn`), parameter footprint profiling (0.18% trainable parameters), and causal language modeling loss backprop on Apple Silicon MPS. |

---

## Mathematical Formulation

### 1. Low-Rank Decomposition
Traditional fine-tuning updates all parameters in a weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$W = W_0 + \Delta W$$

LoRA hypothesizes that weight updates have a low "intrinsic dimension". It decomposes the update matrix $\Delta W$ into two low-rank matrices:

$$\Delta W = B \cdot A$$

Where:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized via Gaussian / Kaiming distribution $\mathcal{N}(0, \sigma^2)$
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to $0$
- Rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$

### 2. Forward Propagation with Scaling
For a given input vector $x$, the forward computation decomposes as:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A x)$$

Where:
- $\alpha$ is a constant scaling hyperparameter.
- Setting $B = 0$ at initialization ensures $\Delta W = 0$ at training step 0, preserving the base pre-trained model's original predictions.

### 3. Parameter Efficiency & Memory Dynamics
Given a square linear transformation matrix of dimension $d$:
- **Full Fine-Tuning Parameters:** $d \times d = d^2$
- **LoRA Adapter Parameters:** $2 \times d \times r$
- **Parameter Reduction Ratio:** $\frac{2 r}{d}$

For example, when $d = 4096$ and rank $r = 8$, LoRA requires only:

$$\frac{2 \times 8}{4096} = \frac{16}{4096} \approx 0.39\% \text{ of the original parameters}$$

During training, optimizer states (Adam momentum and variance buffers) are only allocated for $2dr$ parameters, reducing GPU VRAM allocation.

---

## Verification & Execution

Run the verification scripts from the repository root:

```bash
# 1. Verify LoRA math, B=0 preservation, and autograd isolation
python 12_PEFT_and_LoRA/01_lora_linear_scratch.py

# 2. Verify dynamic adapter injection on Attention blocks
python 12_PEFT_and_LoRA/02_peft_lora_finetune.py

# 3. Verify Hugging Face PEFT LoRA integration and footprint
python 12_PEFT_and_LoRA/03_hf_peft_lora_finetune.py