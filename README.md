# 🚀 journey-to-job-ai-ml

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)%20%7C%20CUDA-green.svg)](https://pytorch.org/docs/stable/notes/mps.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A structured repository tracking my engineering journey from core programming, data structures, and mathematical foundations to production Deep Learning, Modern Large Language Models (LLMs), RAG systems, and Post-Training Quantization.

---

## 📂 Curriculum Architecture

| Phase | Directory | Core Topics & Mechanics |
| :--- | :--- | :--- |
| **Foundations** | [`01_Python_Basics`](./01_Python_Basics) | Modules, package management, control flow, functions, and scopes. |
| | [`02_Data_Structures`](./02_Data_Structures) | Lists, tuples, hash tables, stacks, queues, linked lists, trees, and graphs. |
| | [`03_Algorithms_and_Patterns`](./03_Algorithms_and_Patterns) | Two pointers, sliding window, binary search, recursion, greedy, and dynamic programming. |
| | [`04_Problem_Solving`](./04_Problem_Solving) | Algorithmic implementations: Two Sum, 3Sum, Valid Anagram, Binary Search. |
| **Mathematics & Classical ML** | [`05_Math_for_ML`](./05_Math_for_ML) | Vector operations, matrix multiplication, numerical derivatives, probability & statistics. |
| | [`06_Classical_ML`](./06_Classical_ML) | Linear & Logistic Regression, KNN, K-Means clustering, Decision Trees. |
| **Deep Learning & Computer Vision** | [`07_Deep_Learning_Basics`](./07_Deep_Learning_Basics) | Computational graphs, autograd engine, activation functions, custom MLP, DataLoader pipelines, checkpointing. |
| | [`08_Computer_Vision_CNNs`](./08_Computer_Vision_CNNs) | 2D Convolutions from scratch, CNN architectures, data augmentations, training loops. |
| **NLP, Transformers & GenAI** | [`09_NLP_and_Sequence_Models`](./09_NLP_and_Sequence_Models) | Vocab building, tokenization, PyTorch embeddings, LSTM sentiment classifier, scaled dot-product attention. |
| | [`10_Transformers_and_Attention`](./10_Transformers_and_Attention) | Positional encodings, Multi-Head Attention, Transformer Encoder block from scratch. |
| | [`11_Generative_AI_and_Decoders`](./11_Generative_AI_and_Decoders) | Causal decoder block, mini-GPT causal language model, Hugging Face generation pipelines. |
| **LLM Specialization & Post-Training** | [`12_PEFT_and_LoRA`](./12_PEFT_and_LoRA) | Low-Rank Adaptation (LoRA) linear layers from scratch, PEFT LoRA fine-tuning workflows. |
| | [`13_RAG_and_Vector_Search`](./13_RAG_and_Vector_Search) | Dense embeddings, vector chunking, retrieval indexing, end-to-end grounded RAG with LLMs. |
| | [`14_LLM_Alignment_and_DPO`](./14_LLM_Alignment_and_DPO) | Bradley-Terry preference modeling, Direct Preference Optimization (DPO) token log-probs, Hugging Face TRL DPOTrainer. |
| | [`15_Model_Quantization_and_Compression`](./15_Model_Quantization_and_Compression) | Uniform asymmetric/symmetric quantization, block-wise AbsMax outlier mitigation, in-memory INT8 layers, BitsAndBytes configs. |

---

## 🛠️ Environment & Hardware Acceleration

All deep learning and LLM scripts are configured to dynamically leverage hardware accelerators with automatic fallbacks:
- **Apple Silicon**: Metal Performance Shaders (`mps`) via `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **NVIDIA GPU**: CUDA acceleration.
- **CPU**: Vectorized execution fallback.

```bash
# Clone the repository
git clone [https://github.com/Rebel7363/journey-to-job-ai-ml.git](https://github.com/Rebel7363/journey-to-job-ai-ml.git)
cd journey-to-job-ai-ml

# Activate virtual environment
source .venv/bin/activate

# Install locked dependencies
pip install -r requirements.txt