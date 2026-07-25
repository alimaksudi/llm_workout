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

## 📖 Glossary

Every technical term used anywhere in this course, defined in plain English:
**[GLOSSARY.md](./GLOSSARY.md)**. (Math *symbols* — Σ, ∇, log, sin, Greek letters — are
taught from zero in notebook 03, the math primer.)

## 🗺️ Roadmap & Progress Tracker

> The full learning path — with prerequisites and learning outcomes for every
> module — lives in **[CURRICULUM.md](./CURRICULUM.md)**. Work through the
> notebooks in order; each ends with a "🏋️ Try it yourself" exercise.
>
> Notebooks are grouped into **numbered topic folders** under `notebooks/`
> (`00_prerequisites/`, `01_ml_foundations/`, `02_text_to_vectors/`, … up to
> `10_applied/`). The two-digit file-number prefix is global, so reading them in
> numeric order across folders is the intended sequence.

### Part 0: Foundations & Setup
- [x] **01. Setup & a PyTorch Crash Course**: Tensors, broadcasting, matmul, autograd.
- [x] **02. Neural Networks in 30 Minutes**: Neurons, parameters, gradient descent, training a tiny net.

### Part 1: Mathematical Building Blocks
- [x] **03. The Math You'll Actually Need**: A from-zero primer — reading formulas & symbols, Σ, exponentials & *e*, logarithms, vectors (length/angle/cosine), mean & variance, derivatives/slopes, probability & expectation, **Greek letters & remaining notation (∇, ᵀ, ln)**, **sine/cosine waves**. Every symbol the course uses, with pictures.
- [x] **04. Tensors & Linear Algebra**: Matrix operations, dimensionality tracking, dot products, the row×column matmul mechanic, softmax step by step.
- [x] **05. Probability & Calculus**: Softmax, cross-entropy, gradients.
- [x] **06. Backpropagation From Scratch**: The chain rule by hand in raw NumPy, finite-difference checks, SGD & Adam from scratch, verified against autograd.

### Part 2: From Text to Vectors
- [x] **07. Tokenization & Embeddings**: Build BPE from scratch, then turn text into vectors (embedding space).
- [x] **08. Positional Encoding**: Injecting sequence order (Sin/Cos vs. RoPE — from the 2-D toy to the complex-multiply implementation).

### Part 3: The Attention Engine
- [x] **09. Attention Mechanisms**: Scaled dot-product, self-attention, multi-head, causal masking.

### Part 4: Assembling the Transformer
- [x] **10. The Encoder Layer**: Residual connections, RMSNorm, the FFN — and SwiGLU built from scratch.
- [x] **11. The Decoder Layer**: Masked attention and (historical) cross-attention.
- [x] **12. The Full Transformer**: Putting it all together (decoder-only).
- [x] **13. Assemble Your Decoder-Only GPT**: Wire the real components into the working GPT, learn weight tying, and prove it equals the library `GPT`.

### Part 5: Training Your Model
- [x] **14. Next-Token Prediction & Cross-Entropy Loss**: How wrongness is measured.
- [x] **15. The Training Loop**: AdamW, backprop, LR schedules (warmup + cosine), and what big runs add.
- [x] **16. Train Your Own GPT (Capstone)**: Train the model you built, end to end — with a proof that the library block equals your notebook code.
- [x] **17. When Training Goes Wrong**: A debugging clinic — four classic failures, their fingerprints, and the one-line checks that diagnose them.
- [x] **18. Decoding & Sampling**: Greedy, temperature, top-k, top-p, repetition penalty.
- [x] **19. Evaluating a Language Model**: Perplexity, held-out loss, benchmarks.

### Part 6: Fine-Tuning & Alignment
- [x] **20. Supervised Fine-Tuning (SFT)**: Turning a base model into a chatbot (conceptual).
- [x] **21. Parameter-Efficient Fine-Tuning (LoRA)**: Cheap fine-tuning with low-rank adapters — hands-on, on your own capstone checkpoint.
- [x] **22. Preference Alignment (DPO)**: Aligning to human preferences (conceptual).

### Part 7: Inference Optimization
- [x] **23. KV Caching**: Optimizing autoregressive generation speed.
- [x] **24. Advanced Attention**: GQA (Grouped Query) and MQA (Multi-Query).
- [x] **25. FlashAttention**: Memory tiling and hardware-aware attention.

### Part 8: Scaling the Architecture
- [x] **26. Mixture of Experts (MoE)**: Sparse routing for scaling.
- [x] **27. Scaling Laws**: Why params × data × compute work, and Chinchilla-optimal budgets.

### Part 9: Production & Serving
- [x] **28. Quantization Fundamentals**: Absmax, zero-point, and memory savings.
- [x] **29. Speculative Decoding**: Fast inference with draft vs. target models.
- [x] **30. PagedAttention & Continuous Batching**: vLLM-style serving.
- [x] **31. Graduation — Reading a Real LLM**: Map your library onto Llama-3 and GPT-2, count Llama-3-8B's parameters by hand, and walk a reading list of real model source.

---

## Part II: Applied LLM Engineering

> You built the model. Now build systems that *use* it — prompting, retrieval, RAG, evaluation.
> Requires `pip install -e ".[applied]"`. See [CURRICULUM_PART_II.md](./CURRICULUM_PART_II.md) for the full blueprint.

### Part A: Using LLMs in Practice
- [x] **32. Calling a Model**: Chat interface, roles, tokens, temperature, streaming.
- [x] **33. Structured Output**: JSON prompting, parsing, schema validation, retries, Pydantic.

### Part B: Prompt Engineering
- [x] **34. Prompt Engineering**: Zero/few-shot, chain-of-thought, self-consistency, reliability, injection defense.

### Part C: Embeddings & Semantic Search
- [x] **35. Embeddings & Semantic Search**: Sentence embeddings, cosine similarity, brute-force vector index, k-means, PCA visualization.

### Part D: Retrieval-Augmented Generation (RAG)
- [x] **36. RAG — Retrieval**: Why RAG, chunking strategies, dense search, BM25 from scratch, hybrid RRF retrieval.
- [x] **37. RAG — Generation (Capstone)**: Cross-encoder reranking, context construction, lost-in-the-middle, citations, end-to-end RAG system.

### Part E: Evaluating LLM Systems
- [x] **38. Evaluating LLM Systems**: Exact match, token F1, semantic similarity, faithfulness, LLM-as-judge, hallucination detection, regression testing.

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
