import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    """
    Extracts sentence-level embeddings by averaging token vectors
    while ignoring padding tokens.
    """
    token_embeddings = model_output[0]  # First element contains hidden-states
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def cosine_similarity_scratch(query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity from first principles:
    sim(u, v) = (u . v) / (||u|| * ||v||)
    """
    # L2 normalize vectors along dimension -1
    query_norm = query_emb / torch.norm(query_emb, p=2, dim=-1, keepdim=True)
    doc_norms = doc_embs / torch.norm(doc_embs, p=2, dim=-1, keepdim=True)

    # Matrix multiplication of normalized vectors yields exact cosine similarity
    scores = torch.matmul(doc_norms, query_norm.transpose(-2, -1)).squeeze(-1)
    return scores


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Dense Embeddings & Cosine Similarity Verification on {device} ===\n")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    corpus = [
        "PyTorch offers automatic differentiation and dynamic computation graphs.",
        "Convolutional neural networks are specialized for processing grid-structured visual data.",
        "Retrieval-Augmented Generation grounds large language models with external vector search.",
        "Supervised learning trains models on input-output feature pairs.",
    ]

    query = "How do we ground generative models using retrieved document context?"

    print(f"Query: '{query}'\n")
    print("Corpus Documents:")
    for idx, doc in enumerate(corpus):
        print(f"  [{idx}] {doc}")

    # 1. Encode query and corpus
    all_texts = [query] + corpus
    encoded_input = tokenizer(
        all_texts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        model_output = encoder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    query_embedding = sentence_embeddings[0:1]  # (1, d_emb)
    corpus_embeddings = sentence_embeddings[1:]  # (N, d_emb)

    # 2. Similarity calculation
    similarity_scores = cosine_similarity_scratch(query_embedding, corpus_embeddings)
    ranked_indices = torch.argsort(similarity_scores, descending=True)

    print("\n--- Semantic Similarity Ranking ---")
    for rank, idx in enumerate(ranked_indices.tolist(), start=1):
        score = similarity_scores[idx].item()
        print(f"Rank {rank} (Score: {score:.4f}) -> Doc [{idx}]: {corpus[idx]}")

    # Top match check (Document 2 discusses RAG and grounding)
    top_doc_idx = ranked_indices[0].item()
    assert top_doc_idx == 2, f"Expected Doc 2 as top semantic match, got {top_doc_idx}"
    print("\nVerification Passed: Dense semantic search correctly identified the top context document.")


if __name__ == "__main__":
    main()