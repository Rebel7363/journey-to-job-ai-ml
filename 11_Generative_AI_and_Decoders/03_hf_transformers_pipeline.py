import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def demonstrate_tokenization_and_forward(model_name: str, device: str):
    print(f"\n--- 1. Low-Level Tokenizer & Model Forward Pass ({model_name}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    prompt = "Deep learning and Transformers revolutionized"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"Prompt: '{prompt}'")
    print(f"Encoded Token IDs: {inputs['input_ids'].tolist()[0]}")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (Batch, Seq_Len, Vocab_Size)

    print(f"Logits Tensor Shape: {tuple(logits.shape)}")

    # Next token prediction check
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    next_word = tokenizer.decode([next_token_id])
    print(f"Predicted immediate next token: '{next_word.strip()}' (ID: {next_token_id})")


def demonstrate_pipeline_generation(model_name: str, device: str):
    print(f"\n--- 2. High-Level Text Generation Pipeline ---")
    # device_idx: 0 for CUDA/MPS or -1 for CPU
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=0 if device in ["cuda", "mps"] and torch.cuda.is_available() else -1
    )

    prompt = "Neural networks optimize parameters by"
    results = generator(
        prompt,
        max_new_tokens=30,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    print(f"Prompt:     '{prompt}'")
    print(f"Generated:  '{results[0]['generated_text'].strip()}'")


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Hugging Face Pre-Trained Transformers Pipeline on {device} ===")

    model_name = "distilbert/distilgpt2"
    demonstrate_tokenization_and_forward(model_name, device)
    demonstrate_pipeline_generation(model_name, device)

    print("\nVerification Passed: Tokenizer, logits generation, and pipeline inference executed cleanly.")


if __name__ == "__main__":
    main()