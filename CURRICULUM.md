# 🎓 LLM Workout — Curriculum

> **The learning journey, end to end.** This document is the blueprint for the
> course: the order lessons should be taught in, what each one assumes, what it
> teaches, and what a student can *do* afterward. It is the source of truth for
> sequencing — the notebook file numbers follow it, not the other way around.

## What this course is (and isn't)

**This is a "build a Large Language Model from scratch" course.** You start from
tensors and finish with a trained, aligned, optimized, served decoder-only LLM —
understanding every component because you built it yourself.

There are two kinds of "AI Engineer":

| | **Model-builder** (this course) | **Application-builder** (a separate track) |
| --- | --- | --- |
| Goal | Build & optimize the model itself | Build systems that *use* models |
| Topics | architecture, training, alignment, inference systems | prompting, embeddings/search, RAG, evaluation, agents, serving apps |

This course is the **model-builder** track. Applied topics (RAG, agents, prompt
engineering, production app serving) are intentionally **out of scope** and listed
in [Appendix B](#appendix-b--out-of-scope-a-future-applied-track) so they aren't
mistaken for gaps.

## Who it's for

A motivated programmer who knows **basic Python** and wants to understand how
modern LLMs (Llama-3 style) actually work, by building one. Part 0 provides the
math/ML on-ramp so no prior deep-learning experience is required.

## How to use it

Each module has a theory chapter (`docs/`) and a hands-on notebook
(`notebooks/`). **Read the chapter, then do the notebook.** Every notebook ends
with a "🏋️ Try it yourself" exercise — do it before moving on.

---

## The learning path

Legend: ✅ exists today · 🔵 exists, **moves** in the new order · 🟢 **new** content to write

### Part 0 — Foundations & Setup `🟢 new`
*Goal: everyone starts on the same floor.*

| # | Module | Prerequisites | You will learn | After this you can | Status |
|---|--------|---------------|----------------|--------------------|--------|
| 0.0 | **Prologue: the world before Transformers** | none | why RNNs failed, what attention unlocked | explain *why* LLMs are built the way they are | ✅ (`docs/00`) |
| 0.1 | **Setup & PyTorch crash course** | basic Python | venv, install, tensors, autograd basics, running notebooks | run every notebook in this repo; manipulate tensors | 🟢 new |
| 0.2 | **Neural networks in 30 minutes** | 0.1 | neuron, parameter, forward pass, loss, gradient descent, train/val split | hold the mental model NB 1.x currently *assumes* | 🟢 new |

> **Why new:** today the first notebook opens on dot products and assumes
> PyTorch + "what a neural net is." Beginners hit a wall on page one.

### Part 1 — Mathematical Building Blocks
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 1.1 | **Tensors & linear algebra** | 0.2 | tensors, matmul, dot product = alignment | track shapes; reason about matrix ops | ✅ (`NB01`) |
| 1.2 | **Probability & calculus** | 1.1 | softmax, cross-entropy preview, gradients/chain rule | understand how learning signal flows | ✅ (`NB02`) |

### Part 2 — From Text to Vectors
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 2.1 | **Tokenization & embeddings** | 1.2 | BPE sub-words, embedding lookup, semantic space (cosine) | turn text into vectors; explain why integers won't do | ✅ (`NB03`) |
| 2.2 | **Positional encoding** | 2.1 | sinusoidal vs RoPE, why order must be injected | give a model a sense of sequence order | ✅ (`NB04`) |

### Part 3 — The Attention Engine
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 3.1 | **Attention mechanisms** | 2.2 | Q/K/V, scaled dot-product, multi-head, causal masking | implement self-attention; read an attention matrix | ✅ (`NB05`) |

### Part 4 — Assembling the Transformer
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 4.1 | **The encoder layer** | 3.1 | residuals, normalization (RMSNorm), FFN | build & stack a stable transformer block | ✅ (`NB06`) |
| 4.2 | **The decoder layer** | 4.1 | masked self-attention, cross-attention (historical) | understand the 2017 decoder | ✅ (`NB07`) |
| 4.3 | **The full transformer** | 4.2 | end-to-end assembly, decoder-only paradigm | assemble a complete model | ✅ (`NB08`) |

### Part 5 — Training Your Model `🔵 moved earlier`
*Goal: make the model you just built actually learn. This is the heart of the course and currently comes far too late.*

| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 5.1 | **Next-token prediction & loss** | 4.3 | cross-entropy as "surprise", shifted targets | quantify how wrong a prediction is | 🔵 moved (`NB11`) |
| 5.2 | **The training loop** | 5.1 | AdamW, backprop, zero_grad, epochs | write a full training loop | 🔵 moved (`NB12`) |
| 5.3 | **Capstone — train your own GPT** | 5.2 | train on TinyShakespeare, watch loss fall, sample text | **train a real model from scratch end-to-end** | 🟢 new (promote `scripts/train_tiny.py`) |
| 5.4 | **Decoding & sampling** | 5.3 | greedy vs sampling, **temperature, top-k, top-p**, repetition penalty | control generation quality/creativity | 🟢 new |
| 5.5 | **Evaluating a language model** | 5.3 | perplexity, held-out loss, qualitative checks, benchmark intuition | **measure** whether a model is good | 🟢 new |

> **Why moved:** modules 5.1–5.2 (loss + training) currently sit at notebooks
> 11–12, *after* the model is built **and** after inference optimizations — so
> students "generate" from an untrained model and learn optimization before they
> know what a loss is. Training belongs immediately after assembly.

### Part 6 — Adapting Models: Fine-Tuning & Alignment
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 6.1 | **Supervised fine-tuning (SFT)** | 5.5 | base→chat, ChatML, loss masking | turn a base model into an instruction-follower | ✅ (`NB13`) |
| 6.2 | **Parameter-efficient fine-tuning (LoRA)** | 6.1 | low-rank updates, why ΔW is low-rank, adapters | fine-tune cheaply — *the way it's really done* | 🔵 moved (`NB15`) |
| 6.3 | **Preference alignment (DPO)** | 6.2 | preference pairs, reference model, the DPO loss | align a model to human preferences | ✅ (`NB14`) |

> **Why moved:** LoRA (NB15) currently comes *after* DPO, so students learn full
> fine-tuning, then alignment, then finally "oh, here's the cheap way." LoRA
> belongs right after SFT, framed as the default method.

### Part 7 — Inference Optimization `🔵 moved later`
*Goal: now that you can train, make generation fast and cheap.*

| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 7.1 | **KV caching** | 5.4 | cache past keys/values, identical output, big speedup | make autoregressive generation fast | 🔵 moved (`NB09`) |
| 7.2 | **Advanced attention: MQA & GQA** | 7.1 | shrinking the KV cache, the heads/quality tradeoff | reason about memory vs quality | 🔵 moved (`NB10`) |
| 7.3 | **FlashAttention** | 7.1 | tiling, online softmax, HBM vs SRAM | explain hardware-aware attention | 🔵 moved (`NB17`) |

> **Why moved:** KV caching & MQA/GQA (NB09–10) are *inference-time* concerns
> currently taught before training even exists. They now follow Part 5.

### Part 8 — Scaling the Architecture
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 8.1 | **Mixture of Experts (MoE)** | 7.x | sparse routing, top-k experts, load balancing | explain sparse scaling (Mixtral-style) | ✅ (`NB16`) |
| 8.2 | **Scaling laws** | 8.1 | params × data × compute, Chinchilla-optimal budgets | reason about *why* bigger+more-data helps and how to spend compute | ✅ (`NB23`) |

### Part 9 — Production & Serving
*Goal: the systems that make serving an LLM viable. (Simulations, not a full server — see scope note.)*

| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 9.1 | **Quantization** | 7.x | absmax, zero-point, INT8, memory savings | shrink a model for deployment | ✅ (`NB18`) |
| 9.2 | **Speculative decoding** | 7.1 | draft+verify, accept/reject, the speedup | explain fast multi-token generation | ✅ (`NB19`) |
| 9.3 | **PagedAttention & continuous batching** | 7.1 | block tables, memory paging, batching | explain vLLM-style serving + capstone synthesis | ✅ (`NB20`) |

---

## Appendix A — Module → notebook file map

The notebook files are numbered to follow this curriculum order.

| Module | Topic | Notebook file |
| --- | --- | --- |
| 0.0 | Prologue | `docs/00_prologue.md` |
| 0.1 | Setup & PyTorch crash course | `notebooks/01_setup_and_pytorch.ipynb` |
| 0.2 | Neural networks in 30 min | `notebooks/02_neural_networks_in_30_min.ipynb` |
| 1.1 | Tensors & linear algebra | `notebooks/03_math_foundations.ipynb` |
| 1.2 | Probability & calculus | `notebooks/04_probability_and_calculus.ipynb` |
| 2.1 | Tokenization & embeddings | `notebooks/05_tokenization_and_embeddings.ipynb` |
| 2.2 | Positional encoding | `notebooks/06_positional_encoding.ipynb` |
| 3.1 | Attention | `notebooks/07_attention_mechanisms.ipynb` |
| 4.1 | Encoder layer | `notebooks/08_the_encoder_layer.ipynb` |
| 4.2 | Decoder layer | `notebooks/09_the_decoder_layer.ipynb` |
| 4.3 | Full transformer | `notebooks/10_full_transformer.ipynb` |
| 5.1 | Loss | `notebooks/11_cross_entropy_loss.ipynb` |
| 5.2 | Training loop | `notebooks/12_the_training_loop.ipynb` |
| 5.3 | Capstone: train your GPT | `notebooks/13_train_your_own_gpt.ipynb` |
| 5.4 | Decoding & sampling | `notebooks/14_decoding_and_sampling.ipynb` |
| 5.5 | Evaluation | `notebooks/15_evaluating_a_language_model.ipynb` |
| 6.1 | SFT | `notebooks/16_supervised_fine_tuning.ipynb` |
| 6.2 | LoRA | `notebooks/17_peft_and_lora.ipynb` |
| 6.3 | DPO | `notebooks/18_dpo_preference_alignment.ipynb` |
| 7.1 | KV caching | `notebooks/19_kv_caching.ipynb` |
| 7.2 | MQA/GQA | `notebooks/20_advanced_attention.ipynb` |
| 7.3 | FlashAttention | `notebooks/21_flash_attention.ipynb` |
| 8.1 | MoE | `notebooks/22_mixture_of_experts.ipynb` |
| 8.2 | Scaling laws | `notebooks/23_scaling_laws.ipynb` |
| 9.1 | Quantization | `notebooks/24_quantization_fundamentals.ipynb` |
| 9.2 | Speculative decoding | `notebooks/25_speculative_decoding.ipynb` |
| 9.3 | PagedAttention | `notebooks/26_paged_attention.ipynb` |

## Appendix B — Out of scope (a future "Applied LLM" track)

Deliberately **not** covered here; these belong to an application-builder course:
prompt engineering (zero/few-shot, chain-of-thought, structured output) ·
embeddings for **semantic search & clustering** · **RAG** (chunking, retrieval,
hybrid search, reranking, context construction) · **evaluation** of generations
(hallucination, groundedness, faithfulness, relevance) · **agents** (tool calling,
planning, memory, multi-agent) · **production apps** (response caching, monitoring,
observability, guardrails, fallbacks, cost/latency budgeting).

Listed so they're recognized as a *separate scope decision*, not oversights.
