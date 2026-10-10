import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Hugging Face PEFT LoRA Integration on {device} ===\n")

    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Base parameters count
    total_base_params = sum(p.numel() for p in base_model.parameters())

    # 1. Configure LoRA adapter targeting attention projections
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],  # DistilGPT2 combined Q, K, V projection
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 2. Wrap base model with PEFT LoRA
    peft_model = get_peft_model(base_model, lora_config).to(device)

    print("--- Model Parameter Footprint ---")
    peft_model.print_trainable_parameters()

    # 3. Quick forward pass & loss computation check
    prompt = "LoRA parameter-efficient fine-tuning works by"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs["labels"] = inputs["input_ids"].clone()

    peft_model.train()
    outputs = peft_model(**inputs)
    loss = outputs.loss

    print(f"\nForward verification loss: {loss.item():.4f}")

    # Backprop verification: check that base weights are untouched
    loss.backward()
    trainable_grads = sum(
        1 for p in peft_model.parameters() if p.grad is not None and p.requires_grad
    )
    frozen_grads = sum(
        1 for p in peft_model.parameters() if p.grad is not None and not p.requires_grad
    )

    assert frozen_grads == 0, "Base parameters unexpectedly received gradients!"
    print(f"Gradient Verification Passed: {trainable_grads} adapter tensors updated cleanly.")


if __name__ == "__main__":
    main()