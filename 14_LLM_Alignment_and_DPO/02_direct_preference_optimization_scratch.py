import os
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniLanguageModel(nn.Module):
    """
    Lightweight Causal LM architecture for policy verification.
    """

    def __init__(self, vocab_size: int = 500, d_model: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=128, batch_first=True
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        h = self.transformer(x)
        return self.lm_head(h)


def compute_sequence_log_probs(
    model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Computes sum of log-likelihoods for target tokens:
    sum_t log P(y_t | y_<t, x)
    """
    logits = model(input_ids)
    # Shift logits and labels for autoregressive next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Log softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-probabilities of actual target tokens
    gathered_log_probs = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Sum log probabilities over sequence tokens
    return gathered_log_probs.sum(dim=-1)


def dpo_loss_from_scratch(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimization (DPO) objective from mathematical first principles:
    L_DPO = -log sigmoid( beta * [log(pi(yw)/pi_ref(yw)) - log(pi(yl)/pi_ref(yl))] )
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)

    # Numerically stable -log(sigmoid(x)) = softplus(-x)
    losses = F.softplus(-logits)
    return losses.mean()


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"=== Direct Preference Optimization (DPO) Verification on {device} ===\n")

    torch.manual_seed(42)
    policy_model = MiniLanguageModel().to(device)

    # Reference policy is a frozen copy of the initial model
    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-3)

    # Synthetic chosen vs rejected sequence batches
    batch_size, seq_len = 6, 10
    chosen_ids = torch.randint(1, 499, (batch_size, seq_len), device=device)
    rejected_ids = torch.randint(1, 499, (batch_size, seq_len), device=device)

    # Step 0 baseline evaluation
    with torch.no_grad():
        ref_chosen_logps = compute_sequence_log_probs(reference_model, chosen_ids, chosen_ids)
        ref_rejected_logps = compute_sequence_log_probs(reference_model, rejected_ids, rejected_ids)

        init_pi_chosen = compute_sequence_log_probs(policy_model, chosen_ids, chosen_ids)
        init_pi_rejected = compute_sequence_log_probs(policy_model, rejected_ids, rejected_ids)
        init_loss = dpo_loss_from_scratch(
            init_pi_chosen, init_pi_rejected, ref_chosen_logps, ref_rejected_logps, beta=0.1
        )

    print(f"Initial DPO Loss: {init_loss.item():.4f}")

    # Run DPO optimization loop
    policy_model.train()
    for step in range(1, 26):
        optimizer.zero_grad()

        pi_chosen_logps = compute_sequence_log_probs(policy_model, chosen_ids, chosen_ids)
        pi_rejected_logps = compute_sequence_log_probs(policy_model, rejected_ids, rejected_ids)

        loss = dpo_loss_from_scratch(
            pi_chosen_logps,
            pi_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=0.1,
        )
        loss.backward()
        optimizer.step()

    # Final evaluation
    with torch.no_grad():
        final_pi_chosen = compute_sequence_log_probs(policy_model, chosen_ids, chosen_ids)
        final_pi_rejected = compute_sequence_log_probs(policy_model, rejected_ids, rejected_ids)
        final_loss = dpo_loss_from_scratch(
            final_pi_chosen, final_pi_rejected, ref_chosen_logps, ref_rejected_logps, beta=0.1
        )

    implicit_chosen_reward = 0.1 * (final_pi_chosen - ref_chosen_logps)
    implicit_rejected_reward = 0.1 * (final_pi_rejected - ref_rejected_logps)

    print(f"Final DPO Loss:   {final_loss.item():.4f}")
    print(f"Mean Implicit Reward (Chosen):   {implicit_chosen_reward.mean().item():.4f}")
    print(f"Mean Implicit Reward (Rejected): {implicit_rejected_reward.mean().item():.4f}")

    assert final_loss.item() < init_loss.item(), "DPO optimization failed to reduce loss!"
    assert (
        implicit_chosen_reward.mean().item() > implicit_rejected_reward.mean().item()
    ), "Chosen implicit reward not higher than rejected!"
    print("\nVerification Passed: Policy successfully aligned to preferred completions via DPO objective.")


if __name__ == "__main__":
    main()