from dataclasses import dataclass
from typing import List, Tuple
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    embedding: torch.Tensor = None


def chunk_text(text: str, chunk_size: int = 15, overlap: int = 5) -> List[str]:
    """
    Splits text into token/word chunks using a sliding window with overlap.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    assert step > 0, "chunk_size must be strictly greater than overlap"

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


class InMemoryVectorStore:
    """
    Lightweight, high-performance in-memory vector index.
    Stores dense embeddings and retrieves top-k candidates using cosine similarity.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embedding_matrix: torch.Tensor = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        self.chunks.extend(chunks)
        # Stack embeddings into a single 2D tensor: (N_chunks, Embedding_Dim)
        embeddings = [chunk.embedding for chunk in self.chunks]
        self.embedding_matrix = torch.stack(embeddings, dim=0)

        # L2-normalize stored embeddings once during indexing
        norms = torch.norm(self.embedding_matrix, p=2, dim=-1, keepdim=True)
        self.embedding_matrix = self.embedding_matrix / torch.clamp(norms, min=1e-9)

    def similarity_search(
        self, query_emb: torch.Tensor, top_k: int = 2
    ) -> List[Tuple[DocumentChunk, float]]:
        assert self.embedding_matrix is not None, "Vector store is empty."

        # L2-normalize query embedding
        q_norm = query_emb / torch.clamp(torch.norm(query_emb, p=2, dim=-1, keepdim=True), min=1e-9)

        # Cosine similarity via batch dot product: (N_chunks,)
        sim_scores = torch.matmul(self.embedding_matrix, q_norm.squeeze(0))

        # Retrieve top-k scores and indices
        top_k = min(top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(sim_scores, k=top_k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.chunks[idx], score))
        return results


def embed_texts(texts: List[str], tokenizer, model, device: str) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**encoded)
        token_embs = out[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embs.size()).float()
        pooled = torch.sum(token_embs * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return pooled


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Vector Store & Document Chunking Verification on {device} ===\n")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    sample_document = (
        "Artificial Intelligence pipelines combine data preparation, model optimization, and deployment. "
        "Retrieval-Augmented Generation injects factual grounding data into generative responses. "
        "Vector indexes organize continuous vector embeddings for rapid semantic nearest neighbor queries. "
        "Fine-tuning adapts model parameter distributions directly on domain-specific corpora."
    )

    print("--- 1. Document Chunking Demonstration ---")
    raw_chunks = chunk_text(sample_document, chunk_size=14, overlap=4)
    print(f"Total Chunks Created: {len(raw_chunks)}")
    for i, c in enumerate(raw_chunks):
        print(f"  Chunk {i}: '{c}'")

    # 2. Embedding & Indexing Chunks
    print("\n--- 2. Indexing Chunks into Vector Store ---")
    chunk_embeddings = embed_texts(raw_chunks, tokenizer, encoder, device)
    
    vector_store = InMemoryVectorStore()
    doc_chunk_objects = [
        DocumentChunk(chunk_id=i, text=raw_chunks[i], embedding=chunk_embeddings[i])
        for i in range(len(raw_chunks))
    ]
    vector_store.add_chunks(doc_chunk_objects)
    print(f"Successfully indexed {len(vector_store.chunks)} chunks into vector store.")

    # 3. Top-K Similarity Search
    query = "How do vector databases perform fast similarity retrieval?"
    print(f"\n--- 3. Querying: '{query}' ---")
    query_emb = embed_texts([query], tokenizer, encoder, device)
    top_results = vector_store.similarity_search(query_emb, top_k=2)

    for rank, (chunk, score) in enumerate(top_results, start=1):
        print(f"Rank {rank} [Score: {score:.4f}] Chunk ID {chunk.chunk_id}: '{chunk.text}'")

    assert len(top_results) == 2, "Failed to retrieve requested top-k chunks."
    print("\nVerification Passed: Chunking, indexing, and vector similarity search executed cleanly.")


if __name__ == "__main__":
    main()