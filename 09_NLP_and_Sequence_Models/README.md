# Module 09: Natural Language Processing & Sequence Models

This module builds sequence modeling systems from fundamental mathematical principles up to modern attention mechanics. It bridges manual token indexing and recurrent cells with PyTorch's production-grade Bi-LSTMs and Transformer attention layers.

---

## Architecture & Codebase Map

| File | Core Domain | Key Concepts & Mechanics |
| :--- | :--- | :--- |
| `01_tokenization_vocab_scratch.py` | Text Preprocessing | Custom regex tokenizer, vocabulary dictionary builder, special tokens (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`), batch padding |
| `02_pytorch_embeddings_rnn.py` | Recurrent Foundations | Mathematical Vanilla RNN cell from scratch, `nn.Embedding` lookup matrices, sequential dimensional flow via `nn.LSTM` |
| `03_lstm_sentiment_classifier.py` | Sequence Classification | Dynamic variable-length batch collation (`collate_fn`), bidirectional hidden state aggregation, MPS GPU acceleration |
| `04_scaled_dot_product_attention.py` | Modern Transformers | Scaled Dot-Product Attention engine, Softmax affinity scoring, BERT-style bidirectional attention vs GPT-style causal masking |

---

## Core Engineering & Mathematical Principles

### 1. Tokenization & Numerical Mapping
Neural networks cannot process raw strings. Text processing follows a deterministic three-stage ingestion pipe:
1. **Tokenization:** Splitting raw prose into semantic chunks using regex parsing:
   $$\text{"PyTorch is fast"} \longrightarrow [\text{"pytorch"}, \text{"is"}, \text{"fast"}]$$
2. **Numericalization:** Mapping discrete tokens to unique index integers using a curated frequency vocabulary with fallback reserve tokens:
   $$\text{Tokens} \longrightarrow [x_1, x_2, \dots, x_T] \in \mathbb{N}^T$$
3. **Batch Alignment (Collation):** Rectifying variable-length sequences to equal tensor widths via `<PAD>` indices to enable parallel SIMD operations.

### 2. Recurrent State Propagation (Vanilla RNN Cell)
Sequential dependencies are preserved across discrete time steps $t$ via internal memory states:

$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

- $x_t \in \mathbb{R}^{d_{in}}$: Current time step token representation
- $h_{t-1} \in \mathbb{R}^{d_h}$: Hidden context propagated from previous step
- $h_t \in \mathbb{R}^{d_h}$: Updated context representation passed forward

### 3. Bidirectional Context Aggregation (Bi-LSTM)
Standard unidirectional RNNs only encode past context. A bidirectional architecture evaluates sequences along both trajectories:
- $\overrightarrow{h}_t$: Encodes tokens from index $0 \to T$ (Past context)
- $\overleftarrow{h}_t$: Encodes tokens from index $T \to 0$ (Future context)

The terminal representations from both directions are concatenated:
$$h_{context} = [\overrightarrow{h}_T \,\Vert{}\, \overleftarrow{h}_1] \in \mathbb{R}^{2 \cdot d_h}$$

This concatenated context vector is passed through a classification head for robust sequence-level classification.

### 4. Scaled Dot-Product Attention
Attention eliminates the sequential processing bottleneck of RNNs by computing pairwise affinity across all sequence positions simultaneously:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

- **Query ($Q$) & Key ($K$):** Projections used to calculate compatibility scores via matrix dot products.
- **Scaling Factor ($\sqrt{d_k}$):** Counteracts variance explosion for large projection dimensions, preventing vanishing softmax gradients.
- **Masking Matrix ($M$):**
  - **Bidirectional ($M = 0$):** Tokens attend to all positions (used in encoder representations like BERT).
  - **Causal Mask ($M_{i,j} = -\infty \text{ for } j > i$):** Restricts position $i$ to attending only to preceding tokens $j \le i$, enforcing autoregressive generation (used in decoder models like GPT).
- **Value ($V$):** Context vector dynamically aggregated based on normalized softmax weights.

---

## Verification & Execution

Run the complete sequence pipeline scripts:

```bash
# 1. Custom tokenizer & sequence padding
python 09_NLP_and_Sequence_Models/01_tokenization_vocab_scratch.py

# 2. Embedding layer & manual RNN cell
python 09_NLP_and_Sequence_Models/02_pytorch_embeddings_rnn.py

# 3. Bi-LSTM sequence classifier
python 09_NLP_and_Sequence_Models/03_lstm_sentiment_classifier.py

# 4. Scaled Dot-Product Attention
python 09_NLP_and_Sequence_Models/04_scaled_dot_product_attention.py