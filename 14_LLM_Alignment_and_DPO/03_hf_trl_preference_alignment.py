import os
import shutil
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from trl import DPOConfig, DPOTrainer

logging.set_verbosity_error()


def build_synthetic_preference_dataset() -> Dataset:
    data = {
        "prompt": [
            "Explain gradient descent concisely: ",
            "What is LoRA in deep learning? ",
            "Why do we use vector databases in RAG? ",
            "Explain over-fitting in neural networks: ",
        ],
        "chosen": [
            "An optimization algorithm that iteratively steps in the direction of steepest descent to minimize loss.",
            "A parameter-efficient adaptation technique freezing base weights and injecting low-rank decomposed matrices.",
            "To perform fast, scalable semantic nearest-neighbor retrieval using continuous dense vector embeddings.",
            "A phenomenon where a model memorizes training noise and fails to generalize to unseen test distributions.",
        ],
        "rejected": [
            "It is just a mathematical formula used sometimes in computer programming.",
            "LoRA is when you train all model weights from scratch with big GPUs.",
            "Vector databases are regular SQL tables that store text files without indexing.",
            "Overfitting means the network is running too fast on the hardware.",
        ],
    }
    return Dataset.from_dict(data)


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Hugging Face TRL DPO Alignment Pipeline on {device} ===\n")

    model_name = "distilbert/distilgpt2"
    print(f"Loading base policy and tokenizer: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    train_dataset = build_synthetic_preference_dataset()
    print(f"Constructed preference dataset with {len(train_dataset)} pairs.")

    # DPOConfig inherits from TrainingArguments and injects required DPO fields
    training_args = DPOConfig(
        output_dir="./tmp_dpo_output",
        beta=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        max_prompt_length=64,
        max_length=128,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("\nExecuting DPO training step...")
    train_result = dpo_trainer.train()

    final_loss = train_result.training_loss
    print(f"\nDPO Optimization Finished. Training Loss: {final_loss:.4f}")

    assert final_loss is not None and not torch.isnan(torch.tensor(final_loss)), "DPO training yielded NaN loss!"
    print("\nVerification Passed: Hugging Face TRL DPO training loop executed successfully.")

    if os.path.exists("./tmp_dpo_output"):
        shutil.rmtree("./tmp_dpo_output", ignore_errors=True)


if __name__ == "__main__":
    main()