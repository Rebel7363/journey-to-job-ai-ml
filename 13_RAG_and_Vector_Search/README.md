# Module 13: Retrieval-Augmented Generation (RAG) & Vector Search

This module covers the core mechanics of modern Retrieval-Augmented Generation (RAG) and dense vector search pipelines. It implements dense sentence embedding extraction, cosine similarity metrics from scratch, sliding-window chunking, an in-memory vector store, and an end-to-end grounded generation pipeline combining dense retrievers with generative language models.

---

## Technical Implementations

| Script | Domain | Key Architectural Mechanics |
| :--- | :--- | :--- |
| `01_dense_embeddings_and_cosine_search.py` | Embedding Space & Similarity Math | Mean-pooling over token representations with attention masking, mathematical $L_2$ vector normalization, batch cosine similarity computation from scratch, semantic ranking. |
| `02_vector_store_and_chunking.py` | Chunking & Vector Indexing | Sliding-window text chunker with configurable token overlap, `InMemoryVectorStore` indexing, normalized matrix dot-product operations, and Top-$K$ nearest-neighbor retrieval. |
| `03_rag_pipeline_with_llm.py` | End-to-End RAG Systems | Knowledge base ingestion, dense semantic retrieval (`all-MiniLM-L6-v2`), prompt augmentation with context injection, causal generation conditioning (`distilgpt2`), and Apple Silicon MPS fallback handling. |

---

## Mathematical Formulation

### 1. Mean Pooling with Attention Masking
Given hidden states $H \in \mathbb{R}^{B \times T \times D}$ from a transformer encoder and attention mask $M \in \{0, 1\}^{B \times T}$:

$$e = \frac{\sum_{t=1}^{T} H_{t} \cdot M_{t}}{\max\left(\sum_{t=1}^{T} M_{t}, \epsilon\right)}$$

Where $\epsilon = 10^{-9}$ prevents division by zero for padded tokens.

### 2. Cosine Similarity & Vector Normalization
The semantic similarity between query embedding vector $u$ and document embedding vector $v$ is computed as the normalized inner product:

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\Vert{}u\Vert{}_2 \Vert{}v\Vert{}_2} = \hat{u} \cdot \hat{v}$$

Where $\hat{u} = \frac{u}{\Vert{}u\Vert{}_2}$ and $\hat{v} = \frac{v}{\Vert{}v\Vert{}_2}$ are unit vectors on the unit hypersphere $\mathbb{S}^{D-1}$.

### 3. Top-$K$ Retrieval via Matrix Dot Product
For an indexed corpus matrix $\hat{V} \in \mathbb{R}^{N \times D}$ of $N$ normalized document chunks and a query unit vector $\hat{u} \in \mathbb{R}^D$:

$$s = \hat{V} \hat{u} \in \mathbb{R}^N$$

The top-$K$ candidates are retrieved via:

$$\text{Top-}K = \operatorname{arg\,top-}K_{i \in \{1, \dots, N\}} (s_i)$$

---

## Verification & Execution

Run the scripts from the repository root:

```bash
# 1. Verify dense embedding extraction and cosine similarity math
python 13_RAG_and_Vector_Search/01_dense_embeddings_and_cosine_search.py

# 2. Verify sliding window chunking and in-memory vector store
python 13_RAG_and_Vector_Search/02_vector_store_and_chunking.py

# 3. Verify end-to-end RAG pipeline with grounded generation
python 13_RAG_and_Vector_Search/03_rag_pipeline_with_llm.py