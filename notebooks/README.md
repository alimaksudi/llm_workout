# 📓 Notebooks — reading order

**Start at [`00_prerequisites/01_setup_and_pytorch.ipynb`](00_prerequisites/01_setup_and_pytorch.ipynb) and walk the numbers.**

Notebooks are grouped into numbered topic folders, but the **two-digit file prefix
is global** — so reading `01 → 02 → 03 → …` straight through the folders is the
intended path. Read the matching `docs/` chapter first when there is one; every
notebook ends with a "🏋️ Try it yourself" exercise. Full learning outcomes and
prerequisites live in [`../CURRICULUM.md`](../CURRICULUM.md).

> Setup once, from the repo root: `pip install -e ".[notebooks]"` (add `".[applied]"`
> for Part II, notebooks 31+). Notebooks are committed with outputs cleared — run
> them to see the plots and results.

## Part I — Build an LLM from scratch

### `00_prerequisites/` — get set up, see the whole picture
- **01. Setup & a PyTorch crash course** — tensors, broadcasting, matmul, batched matmul, autograd.
- **02. Neural networks in 30 minutes** — neuron, parameter, loss, gradient descent; train a tiny net.

### `01_ml_foundations/` — the math and mechanics, built from zero
- **03. The math you'll actually need** — reading formulas & symbols, Σ, exponentials & *e*, logarithms, vectors (length/angle/cosine), mean & variance, derivatives/slopes, probability & expectation, **Greek letters & remaining notation (∇, ᵀ, ln)**, **sine/cosine waves**.
- **04. Tensors & linear algebra** — dot product = similarity, the row×column matmul mechanic, softmax step by step.
- **05. Probability & calculus** — logits, cross-entropy, gradients, the chain rule.
- **06. Backpropagation from scratch** — every gradient by hand in NumPy, checked against finite differences and autograd; SGD & Adam from scratch.

### `02_text_to_vectors/`
- **07. Tokenization & embeddings** — build BPE from scratch, then embed.
- **08. Positional encoding** — sinusoidal vs. RoPE (2-D toy → complex-multiply implementation).

### `03_attention/`
- **09. Attention mechanisms** — scaled dot-product, self-attention, multi-head, causal masking.

### `04_transformer/`
- **10. The encoder layer** — residuals, RMSNorm, the FFN, SwiGLU from scratch.
- **11. The decoder layer** — masked self-attention, (historical) cross-attention.
- **12. The full transformer** — end-to-end assembly (decoder-only).
- **13. Assemble your decoder-only GPT** — wire the real parts into the working GPT, weight tying, and prove it equals the library.

### `05_training/`
- **14. Next-token prediction & cross-entropy loss** — how wrongness is measured.
- **15. The training loop** — AdamW, warmup + cosine schedules, what big runs add.
- **16. Train your own GPT (capstone)** — train the model you built, end to end.
- **17. When training goes wrong** — a debugging clinic: four classic failures and their one-line checks.
- **18. Decoding & sampling** — greedy, temperature, top-k, top-p, repetition penalty.
- **19. Evaluating a language model** — perplexity, held-out loss, benchmarks.

### `06_finetuning/`
- **20. Supervised fine-tuning (SFT)** — base → chat, loss masking (conceptual).
- **21. Parameter-efficient fine-tuning (LoRA)** — hands-on, on your own capstone checkpoint.
- **22. Preference alignment (DPO)** — the DPO loss (conceptual).

### `07_inference/`
- **23. KV caching** — fast autoregressive generation.
- **24. Advanced attention** — MQA & GQA.
- **25. FlashAttention** — tiling and hardware-aware attention.

### `08_scaling/`
- **26. Mixture of Experts (MoE)** — sparse routing.
- **27. Scaling laws** — params × data × compute, Chinchilla-optimal budgets.

### `09_production/`
- **28. Quantization fundamentals** — absmax, zero-point, INT8.
- **29. Speculative decoding** — draft + verify.
- **30. PagedAttention & continuous batching** — vLLM-style serving.
- **31. Graduation — reading a real LLM** — map your library onto Llama-3 / GPT-2; count Llama-3-8B's parameters by hand.

## Part II — Applied LLM engineering
*Needs `pip install -e ".[applied]"`; these load small pretrained models.*

### `10_applied/`
- **32. Calling a model** — chat interface, roles, tokens, streaming.
- **33. Structured output** — JSON, schema validation, retries, Pydantic.
- **34. Prompt engineering** — zero/few-shot, chain-of-thought, self-consistency, injection defense.
- **35. Embeddings & semantic search** — cosine similarity, a brute-force vector index, k-means, PCA.
- **36. RAG — retrieval** — chunking, dense search, BM25 from scratch, hybrid RRF.
- **37. RAG — generation (capstone)** — reranking, context construction, citations, end-to-end.
- **38. Evaluating LLM systems** — exact match, token F1, faithfulness, LLM-as-judge, hallucination detection.
