# 🚀 LLM Workout: From Zero to Transformer Master

A hands-on journey to mastering Transformer architectures, LLMs, and the mathematics that power them. This repository is designed to bridge the gap between academic papers and production-ready implementation.

## 📌 Project Goals
- **Architecture First**: We focus on building the inner workings of Transformers from scratch.
- **Visual Learning**: Every component is accompanied by a Jupyter Notebook and a docs chapter (with Mermaid diagrams) explaining the "Why" behind the tensors.
- **Modern Standards**: We don't just stop at the 2017 paper; we implement Llama-style improvements (RoPE, RMSNorm, SwiGLU).

> **Notebooks teach, the library ships.** The notebooks are deliberately standalone (each re-implements its concept inline so you can run any one in isolation). The `src/llm_workout/` package is the polished, **tested** distillation of those ideas. See [CONTRIBUTING.md](./CONTRIBUTING.md) for how the pieces fit together.

## 📚 Architectural Textbook (Zero to Hero)
For a deep dive into the mathematical concepts, analogies, and strictly non-code architectural theory driving these Jupyter notebooks, refer to our compiled documentation:
- [Chapter 0: Prologue - History and Limitations](./docs/00_prologue.md)
- [Chapter 1: Mathematics and Structural Building Blocks](./docs/01_math_and_building_blocks.md)
- [Chapter 2: Architecture Assembly](./docs/02_architecture_assembly.md)
- [Chapter 3: Efficiency and Scaling](./docs/03_efficiency_and_scaling.md)
- [Chapter 4: Training and Alignment](./docs/04_training_and_alignment.md)
- [Chapter 5: Advanced Hardware & Mixture of Experts (MoE)](./docs/05_advanced_hardware_and_moe.md)
- [Chapter 6: Production and Inference Engineering](./docs/06_production_and_inference.md)

## 🗺️ Roadmap & Progress Tracker

> The full learning path — with prerequisites and learning outcomes for every
> module — lives in **[CURRICULUM.md](./CURRICULUM.md)**. Work through the
> notebooks in order; each ends with a "🏋️ Try it yourself" exercise.

### Part 0: Foundations & Setup
- [x] **01. Setup & a PyTorch Crash Course**: Tensors, broadcasting, matmul, autograd.
- [x] **02. Neural Networks in 30 Minutes**: Neurons, parameters, gradient descent, training a tiny net.

### Part 1: Mathematical Building Blocks
- [x] **03. Tensors & Linear Algebra**: Matrix operations, dimensionality tracking, dot products.
- [x] **04. Probability & Calculus**: Softmax, cross-entropy, gradients.

### Part 2: From Text to Vectors
- [x] **05. Tokenization & Embeddings**: Turning text into vectors (BPE, embedding space).
- [x] **06. Positional Encoding**: Injecting sequence order (Sin/Cos vs. RoPE).

### Part 3: The Attention Engine
- [x] **07. Attention Mechanisms**: Scaled dot-product, self-attention, multi-head, causal masking.

### Part 4: Assembling the Transformer
- [x] **08. The Encoder Layer**: Residual connections, RMSNorm, and the FFN.
- [x] **09. The Decoder Layer**: Masked attention and (historical) cross-attention.
- [x] **10. The Full Transformer**: Putting it all together (decoder-only).

### Part 5: Training Your Model
- [x] **11. Next-Token Prediction & Cross-Entropy Loss**: How wrongness is measured.
- [x] **12. The Training Loop**: AdamW, backprop, the optimization step.
- [x] **13. Train Your Own GPT (Capstone)**: Train the model you built, end to end.
- [x] **14. Decoding & Sampling**: Greedy, temperature, top-k, top-p, repetition penalty.
- [x] **15. Evaluating a Language Model**: Perplexity, held-out loss, benchmarks.

### Part 6: Fine-Tuning & Alignment
- [x] **16. Supervised Fine-Tuning (SFT)**: Turning a base model into a chatbot.
- [x] **17. Parameter-Efficient Fine-Tuning (LoRA)**: Cheap fine-tuning with low-rank adapters.
- [x] **18. Preference Alignment (DPO)**: Aligning to human preferences.

### Part 7: Inference Optimization
- [x] **19. KV Caching**: Optimizing autoregressive generation speed.
- [x] **20. Advanced Attention**: GQA (Grouped Query) and MQA (Multi-Query).
- [x] **21. FlashAttention**: Memory tiling and hardware-aware attention.

### Part 8: Scaling the Architecture
- [x] **22. Mixture of Experts (MoE)**: Sparse routing for scaling.
- [x] **23. Scaling Laws**: Why params × data × compute work, and Chinchilla-optimal budgets.

### Part 9: Production & Serving
- [x] **24. Quantization Fundamentals**: Absmax, zero-point, and memory savings.
- [x] **25. Speculative Decoding**: Fast inference with draft vs. target models.
- [x] **26. PagedAttention & Continuous Batching**: vLLM-style serving.

---

## Part II: Applied LLM Engineering

> You built the model. Now build systems that *use* it — prompting, retrieval, RAG, evaluation.
> Requires `pip install -e ".[applied]"`. See [CURRICULUM_PART_II.md](./CURRICULUM_PART_II.md) for the full blueprint.

### Part A: Using LLMs in Practice
- [x] **27. Calling a Model**: Chat interface, roles, tokens, temperature, streaming.
- [x] **28. Structured Output**: JSON prompting, parsing, schema validation, retries, Pydantic.

### Part B: Prompt Engineering
- [x] **29. Prompt Engineering**: Zero/few-shot, chain-of-thought, self-consistency, reliability, injection defense.

### Part C: Embeddings & Semantic Search
- [x] **30. Embeddings & Semantic Search**: Sentence embeddings, cosine similarity, brute-force vector index, k-means, PCA visualization.

### Part D: Retrieval-Augmented Generation (RAG)
- [x] **31. RAG — Retrieval**: Why RAG, chunking strategies, dense search, BM25 from scratch, hybrid RRF retrieval.
- [x] **32. RAG — Generation (Capstone)**: Cross-encoder reranking, context construction, lost-in-the-middle, citations, end-to-end RAG system.

### Part E: Evaluating LLM Systems
- [x] **33. Evaluating LLM Systems**: Exact match, token F1, semantic similarity, faithfulness, LLM-as-judge, hallucination detection, regression testing.

---

## 🛠️ Setup
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the core library (torch + numpy)
pip install -e .

# ...or pull in everything needed to run the notebooks:
pip install -e ".[notebooks]"

# ...or the test/development tooling:
pip install -e ".[dev]"

# Part II — Applied LLM Engineering (embeddings + local LLM):
pip install -e ".[applied]"
```

## 🚀 Training Demo

We have extracted the core neural network layers from the educational notebooks into a modular, production-ready `src/llm_workout/` python library.

You can import these components directly into your own projects! The architecture uses modernized Llama-3 standards (RMSNorm, SwiGLU):
```python
from llm_workout.model import GPT
from llm_workout.layers import RMSNorm, SwiGLU

# Initialize a custom GPT
model = GPT(vocab_size=50000, d_model=256, num_layers=4, num_heads=8, hidden_dim=1024)
```

You can also train a miniature Transformer from scratch right in your terminal using the TinyShakespeare dataset! This standalone script downloads the data and streams its learning process to your console:

```bash
python scripts/train_tiny.py
```

When training finishes, the script saves a checkpoint to `checkpoints/tiny_shakespeare.pt`. You can then generate fresh Shakespeare from that checkpoint:

```bash
python scripts/generate.py --prompt "ROMEO:" --max-new-tokens 300
```

## 🧪 Testing

The library ships with a Pytest suite that guards the core invariants (shapes, weight tying, and a KV-cache vs. full-forward equivalence check):

```bash
pip install -e ".[dev]"
pytest -q
```

## 🤝 Contributing

Contributions are very welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for the project layout, conventions, and how to run the test suite.

---
*Created with ❤️ for the LLM community.*
