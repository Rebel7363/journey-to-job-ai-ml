# Module 14: LLM Alignment, Preference Optimization & DPO

This module covers the core mechanics of aligning autoregressive Large Language Models (LLMs) with human intent and preference distributions. It implements the Bradley-Terry preference probability framework, pairwise ranking loss functions from scratch, Direct Preference Optimization (DPO) via implicit reward log-likelihood ratios, and production-grade alignment workflows utilizing Hugging Face's `trl` (`DPOTrainer`) on Apple Silicon MPS acceleration.

---

## Technical Implementations

| Script | Domain | Key Architectural Mechanics |
| :--- | :--- | :--- |
| `01_reward_model_and_pairwise_loss.py` | Reward Modeling & Bradley-Terry Math | Sequence representation pooling, scalar utility projection head $r_\theta(x, y)$, numerically stable Softplus pairwise cross-entropy loss, and gradient-based preference separation. |
| `02_direct_preference_optimization_scratch.py` | DPO Formulation from First Principles | Dual policy/reference sequence token gathering, target log-probability accumulation $\sum \log \pi(y_t \mid x, y_{<t})$, implicit reward margin formulation, and reference policy constraint enforcement. |
| `03_hf_trl_preference_alignment.py` | Production Alignment Pipelines | Pairwise dataset ingestion (`prompt`, `chosen`, `rejected`), Hugging Face `trl` `DPOConfig` / `DPOTrainer` orchestration, reference model caching, and training loss verification on Apple Silicon MPS. |

---

## Mathematical Formulation

### 1. Bradley-Terry Preference Model
Under the Bradley-Terry preference framework, the probability that completion $y_w$ (winner/chosen) is preferred over $y_l$ (loser/rejected) given prompt $x$ parameterized by reward model $r_\theta$ is defined as:

$$P(y_w \succ y_l \mid x) = \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right) = \frac{1}{1 + e^{-\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)}}$$

The negative log-likelihood ranking loss optimized across a preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$ is:

$$\mathcal{L}_{\text{Rank}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right] = \mathbb{E}_{(x, y_w, y_l)}\left[\text{softplus}\left(-(r_\theta(x, y_w) - r_\theta(x, y_l))\right)\right]$$

### 2. Direct Preference Optimization (DPO) Derivation
Traditional RLHF optimizes the KL-constrained reward objective:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[r(x, y)\right] - \beta \mathbb{D}_{\text{KL}}\left(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)$$

Rafailov et al. (2023) analytically proved that the closed-form optimal solution yields an exact mapping from the optimal policy $\pi^*$ to the ground-truth reward:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting this implicit reward representation into the Bradley-Terry objective cancels the partition function $Z(x)$, yielding the parameter-free Direct Preference Optimization (DPO) loss:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

Where:
- $\pi_\theta(y \mid x) = \prod_{t=1}^{\vert{}y\vert{}} \pi_\theta(y_t \mid x, y_{<t})$ represents the autoregressive sequence likelihood.
- $\beta$ is a regularization hyperparameter scaling the penalty against policy drift from $\pi_{\text{ref}}$.

---

## Verification & Execution

Execute the validation scripts from the repository root:

```bash
# 1. Verify Bradley-Terry reward modeling and pairwise ranking loss
python 14_LLM_Alignment_and_DPO/01_reward_model_and_pairwise_loss.py

# 2. Verify DPO objective and log-ratio accumulation from scratch
python 14_LLM_Alignment_and_DPO/02_direct_preference_optimization_scratch.py

# 3. Verify Hugging Face TRL DPOTrainer integration
python 14_LLM_Alignment_and_DPO/03_hf_trl_preference_alignment.py