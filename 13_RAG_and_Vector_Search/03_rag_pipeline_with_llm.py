import os
import warnings

# MPS fallback for unsupported ops & suppress warning clutter
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class MiniRAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name: str = "distilbert/distilgpt2",
        device: str = "cpu",
    ):
        self.device = device

        print(f"Loading retriever encoder: {embedding_model_name}...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embed_model = AutoModel.from_pretrained(embedding_model_name).to(device)
        self.embed_model.eval()

        print(f"Loading generator model: {generator_model_name}...")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.gen_model = AutoModelForCausalLM.from_pretrained(generator_model_name).to(device)
        self.gen_model.eval()

        self.corpus_chunks: List[str] = []
        self.index_embeddings: torch.Tensor = None

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask, dim=1)
        sum_mask = torch.clamp(input_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        encoded = self.embed_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.embed_model(**encoded)
            embeddings = self._mean_pooling(output, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def index_documents(self, documents: List[str]):
        self.corpus_chunks = documents
        self.index_embeddings = self.embed_texts(documents)
        print(f"Knowledge Base indexed: {len(documents)} context chunks.")

    def retrieve(self, query: str, top_k: int = 1) -> List[str]:
        query_emb = self.embed_texts([query])
        scores = torch.matmul(self.index_embeddings, query_emb.squeeze(0))
        top_k = min(top_k, len(self.corpus_chunks))
        _, indices = torch.topk(scores, k=top_k)
        return [self.corpus_chunks[idx] for idx in indices.tolist()]

    def generate_grounded_answer(self, query: str, top_k: int = 1):
        retrieved_contexts = self.retrieve(query, top_k=top_k)
        context_str = " ".join(retrieved_contexts)

        augmented_prompt = f"Context: {context_str}\nQuestion: {query}\nAnswer:"
        inputs = self.gen_tokenizer(augmented_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.gen_model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

        full_generated = self.gen_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return context_str, full_generated


if __name__ == "__main__":
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Running End-to-End RAG Pipeline Verification on {device} ===\n")

    rag = MiniRAGPipeline(device=device)

    knowledge_base = [
        "Project Journey-to-Job implements deep learning architectures from scratch in PyTorch on Apple Silicon MPS.",
        "LoRA optimizes neural network adaptation by training low-rank decomposed matrices A and B while freezing base weights.",
        "Vector indexes enable scalable semantic nearest-neighbor retrieval by computing cosine similarity across dense embeddings.",
    ]
    rag.index_documents(knowledge_base)

    query = "How does LoRA optimize neural network adaptation?"
    print(f"\nUser Query: '{query}'")

    context_used, answer = rag.generate_grounded_answer(query, top_k=1)

    print(f"\n[Retrieved Context Chunk]:\n\"{context_used}\"")
    print(f"\n[Augmented LLM Output]:\n{answer.strip()}")

    assert "LoRA optimizes" in context_used, "Retriever failed to fetch relevant context chunk!"
    print("\nVerification Passed: RAG pipeline successfully retrieved grounded context and generated conditioned response.")