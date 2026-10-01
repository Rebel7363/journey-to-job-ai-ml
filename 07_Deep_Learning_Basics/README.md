# Module 07: Deep Learning Basics

This module establishes core neural network foundations by bridging pure mathematical implementations in NumPy with modern production pipelines in PyTorch.

---

## Architecture & Implementation Overview

| Script | Domain | Key Concepts |
| :--- | :--- | :--- |
| `01_computational_graph_autograd.py` | Math / Autograd | Scalar computational DAG, manual topological sort, automatic backpropagation |
| `02_activation_functions.py` | Math / Activations | Sigmoid, Tanh, ReLU, LeakyReLU, Softmax with analytical first-order derivatives |
| `03_mlp_forward_backward.py` | Neural Networks | 2-layer MLP from scratch in NumPy, vectorized cross-entropy, manual gradient descent |
| `04_pytorch_tensors_autograd.py` | PyTorch Core | Tensors, autograd DAG engine, Apple Metal (MPS) device routing, linear regression |
| `05_pytorch_custom_mlp.py` | PyTorch Architecture | Modular `nn.Module` subclassing, ReLU layers, multi-class CrossEntropyLoss, Adam optimizer |
| `06_pytorch_dataloader_pipeline.py` | Data Engineering | Custom `Dataset` subclassing, batching with `DataLoader`, train/val split, evaluation loops |
| `07_pytorch_save_load_checkpoint.py` | Production ML | Serializing model & optimizer `state_dict`, checkpoint loading, inference verification |

---

## Key Takeaways

1. **Analytical vs Automatic Differentiation:**
   - Writing backward passes by hand validates how gradient accumulation happens across weight matrices.
   - PyTorch's computational graph tracks operations dynamic-on-the-fly (`autograd`), abstracting explicit Jacobian-vector products.

2. **Modular Production Standards:**
   - Structuring networks inside `nn.Module` enforces separation between architecture definition (`__init__`) and computation flow (`forward`).
   - Training workflows decouple data ingestion (`DataLoader`) from optimization steps to ensure memory efficiency across mini-batches.
   - Model weights must be preserved via `state_dict` rather than entire serialized objects to maintain architecture decoupled portability.

---

## Verification & Execution

To execute the pipelines using hardware acceleration:

```bash
python 07_Deep_Learning_Basics/06_pytorch_dataloader_pipeline.py
python 07_Deep_Learning_Basics/07_pytorch_save_load_checkpoint.py