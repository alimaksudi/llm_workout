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
engineering, production app serving) are covered by **Part II**
([CURRICULUM_PART_II.md](./CURRICULUM_PART_II.md), notebooks 35–41) and listed in
[Appendix B](#appendix-b--the-applied-track-part-ii) so they aren't mistaken for gaps.

## Who it's for

A motivated programmer who knows **basic Python** and wants to understand how
modern LLMs (Llama-3 style) actually work, by building one. Part 0 provides the
math/ML on-ramp so no prior deep-learning experience is required.

## How to use it

**The notebooks are the course** — each one is self-contained, and every notebook
ends with a "🏋️ Try it yourself" exercise, so work through them in order and do the
exercise before moving on.

The `docs/` folder is a **companion theory textbook**: prose-only chapters, grouped
by theme rather than one-per-module, for readers who like the concepts laid out away
from the code. Where a chapter covers what you're about to build, reading it first
helps — but you never *need* it, and not every module has one.

Two safety nets so you never need outside material: **[GLOSSARY.md](./GLOSSARY.md)**
defines every technical term in plain English, and **Module 1.1** teaches every math
symbol the course uses from zero.

---

## The learning path

Legend: ✅ exists (notebook file numbers follow this order).

### Part 0 — Foundations & Setup
*Goal: everyone starts on the same floor.*

| # | Module | Prerequisites | You will learn | After this you can | Status |
|---|--------|---------------|----------------|--------------------|--------|
| 0.0 | **Prologue: the world before Transformers** | none | why RNNs failed, what attention unlocked | explain *why* LLMs are built the way they are | ✅ (`docs/00`) |
| 0.1 | **Setup & PyTorch crash course** | basic Python | venv, install, tensors, batched matmul, autograd basics, the everyday PyTorch idioms (`unsqueeze`, `no_grad`, `detach`, train/eval mode, buffers) | run every notebook in this repo; manipulate tensors | ✅ (`NB01`) |
| 0.2 | **Neural networks in 30 minutes** | 0.1 | neuron, parameter, forward pass, loss, gradient descent, train/val split | hold the mental model the later modules assume | ✅ (`NB02`) |

### Part 1 — Mathematical Building Blocks
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 1.1 | **The math you'll actually need** | 0.2 | reading formulas & symbols, Σ, exponentials & *e*, logarithms, vectors (length/angle/cosine), mean & variance, derivatives/slopes, probability & expectation, **Greek letters & remaining notation (∇, ᵀ, ln)**, **sine/cosine waves** | *read* any formula in the course; understand every symbol before it's used | ✅ (`NB03`) |
| 1.2 | **Tensors & linear algebra** | 1.1 | tensors, matmul (the row×column mechanic), dot product = alignment, softmax step by step | track shapes; reason about matrix ops | ✅ (`NB04`) |
| 1.3 | **Probability & calculus** | 1.2 | softmax, cross-entropy preview, gradients/chain rule | understand how learning signal flows | ✅ (`NB05`) |
| 1.4 | **Backpropagation from scratch** | 1.3 | finite differences, the chain rule by hand (raw NumPy), SGD & Adam from scratch, verifying against autograd | derive & *verify* every gradient yourself; treat `loss.backward()` as a convenience, not magic | ✅ (`NB06`) |

> **Why 1.1 exists:** several notebooks lean on logs, exponentials, Σ, slopes,
> vectors, and probability — but those were previously *used* without ever being
> *taught*. This primer builds every one from zero, with a picture and a
> one-line Python demo, so no later formula is a wall.
>
> **Why 1.4 exists:** the rest of the course calls `loss.backward()` constantly.
> This module opens that box exactly once — by hand, in NumPy, checked against
> both finite differences and PyTorch — so nothing downstream is a mystery.

### Part 2 — From Text to Vectors
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 2.1 | **Tokenization & embeddings** | 1.3 | **BPE built from scratch** (merge loop), sub-words, embedding lookup, semantic space (cosine) | train a toy BPE tokenizer; turn text into vectors; explain why integers won't do | ✅ (`NB07`) |
| 2.2 | **Positional encoding** | 2.1 | sinusoidal vs RoPE, the 2-D rotation intuition **and** the complex-multiply implementation used by Llama | give a model a sense of sequence order; read `precompute_freqs_cis` in real code | ✅ (`NB08`) |

### Part 3 — The Attention Engine
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 3.1 | **Self-attention** | 2.2 | Q/K/V as learnable projections, scaled dot-product read stage by stage, one head end to end | implement self-attention; read an attention matrix | ✅ (`NB09`) |
| 3.2 | **Multi-head attention & causal masking** | 3.1 | the head reshape walked axis by axis, why heads specialize, causal masking, **the O(T²) bill** | build the attention block real models use; explain what Part 7 exists to fix | ✅ (`NB10`) |

> **Why 3.1 splits into two:** this is the load-bearing notebook of the whole course,
> and it previously carried four distinct ideas (Q/K/V, scaled dot-product, multi-head,
> masking) in one sitting. The `(B,T,C) → (B,T,H,C/H) → (B,H,T,C/H)` reshape defeats
> more learners than any other line in Transformer code; it now gets its own module,
> with every intermediate shape printed.

### Part 4 — Assembling the Transformer
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 4.1 | **The encoder layer** | 3.2 | residuals, normalization (RMSNorm), FFN, **SwiGLU built from scratch** | build & stack a stable transformer block with the modern gated FFN | ✅ (`NB11`) |
| 4.2 | **The decoder layer** | 4.1 | masked self-attention, cross-attention (historical) | understand the 2017 decoder | ✅ (`NB12`) |
| 4.3 | **The full transformer** | 4.2 | end-to-end assembly, decoder-only paradigm | assemble a complete model | ✅ (`NB13`) |
| 4.4 | **Assemble your decoder-only GPT** | 4.3 | wire the real components into the working GPT, weight tying, prove it equals the library | **build the exact model you'll train** — no more importing a black box | ✅ (`NB14`) |

### Part 5 — Training Your Model
*Goal: make the model you just built actually learn. This is the heart of the course.*

| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 5.0 | **From corpus to token stream** | 4.4, 2.1 | sourcing & data mixtures, quality filtering, exact + MinHash dedup, **training your Module 2.1 BPE on a real corpus**, `uint16` streams, packing & EOS boundaries, contiguous splits, **contamination detection** | build the data pipeline that feeds the model — and prove your held-out set is really held out | ✅ (`NB15`) |
| 5.1 | **Next-token prediction & loss** | 5.0 | cross-entropy as "surprise", shifted targets | quantify how wrong a prediction is | ✅ (`NB16`) |
| 5.2 | **The training loop** | 5.1, 1.3 | AdamW, backprop, zero_grad, **warmup + cosine LR schedules**, what big runs add (clipping, accumulation, mixed precision, parallelism) | write a full training loop and read a real one | ✅ (`NB17`) |
| 5.3 | **Capstone — train your own GPT** | 5.2 | train on TinyShakespeare, watch loss fall, sample text; **prove the library block == your notebook code** | **train a real model from scratch end-to-end** | ✅ (`NB18`) |
| 5.4 | **When training goes wrong** | 5.3 | the four classic failures (stuck loss, too-good loss, NaN explosion, shape crash), their fingerprints, the single-batch overfit test | *diagnose* a broken training run instead of staring at it | ✅ (`NB19`) |
| 5.5 | **Decoding & sampling** | 5.3 | greedy vs sampling, **temperature, top-k, top-p**, repetition penalty | control generation quality/creativity | ✅ (`NB20`) |
| 5.6 | **Evaluating a language model** | 5.3, 5.0 | perplexity, held-out loss, qualitative checks, benchmark intuition | **measure** whether a model is good | ✅ (`NB21`) |
| 5.7 | **Scaling laws** | 5.6 | params × data × compute, power laws, Chinchilla-optimal budgets | answer "is my model bad, or is it just small?"; reason about how to spend compute | ✅ (`NB22`) |

> **Why this order:** the data pipeline (5.0) comes first because a model with nothing
> to eat is not a training run — and it's where the BPE tokenizer you built in Module
> 2.1 finally gets used on a real corpus. Loss and the training loop follow immediately
> (you should never "generate" from an untrained model without knowing why it's noise).
> The debugging clinic (5.4) lands right after your first real training run, exactly
> when things start breaking on your own experiments.
>
> **Why scaling laws moved here (it used to be 8.2):** it answers the question the
> learner actually asks at this exact moment — *"is my model bad, or is this just what
> a small model does?"* — and it needs only loss, a trained model, and evaluation, all
> of which you now have. It also reframes everything that follows: Parts 6–9 are all
> answers to "how do we ride the scaling law more cheaply?" Teaching it after MoE had
> the question following its own answer.

### Part 6 — Adapting Models: Fine-Tuning & Alignment
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 6.1 | **Supervised fine-tuning (SFT)** | 5.7 | base→chat, ChatML, loss masking *(conceptual module)* | turn a base model into an instruction-follower | ✅ (`NB23`) |
| 6.2 | **Parameter-efficient fine-tuning (LoRA)** | 6.1 | low-rank updates, why ΔW is low-rank, adapters; **hands-on: LoRA-tune your own capstone checkpoint** | fine-tune cheaply — *the way it's really done* — on a model you trained | ✅ (`NB24`) |
| 6.3 | **Reward models & RLHF** | 6.1 | Bradley–Terry, **training a real reward model on your own capstone checkpoint**, the KL-leashed objective, **reward hacking demonstrated**, why PPO needs four models | explain and read *InstructGPT*; understand where DPO's β and π_ref come from | ✅ (`NB25`) |
| 6.4 | **Preference alignment (DPO)** | 6.3 | preference pairs, reference model, the DPO loss *(conceptual module)* | align a model to human preferences | ✅ (`NB26`) |

> **Why this order:** LoRA sits right after SFT, framed as the default method —
> and it's the hands-on module of the trio: you adapt the actual Module 5.3
> checkpoint and watch its voice change.
>
> **Why 6.3 exists:** the DPO loss is not an independent invention — it is a
> rearrangement of the KL-regularized RLHF objective, with the reward model solved
> away analytically. Taught without that, DPO's two most important symbols (β and
> π_ref) are unmotivated magic, and "simpler than RLHF" carries no information because
> RLHF was never explained. This module builds the thing DPO simplifies: you train an
> actual reward model on your capstone checkpoint, then watch a policy hack its own
> grader. It is also the gateway to *InstructGPT*, *Constitutional AI*, and the
> reasoning-model RL literature.

### Part 7 — Inference Optimization
*Goal: now that you can train, make generation fast and cheap.*

| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 7.1 | **KV caching** | 5.5 | cache past keys/values, identical output, big speedup | make autoregressive generation fast | ✅ (`NB27`) |
| 7.2 | **Advanced attention: MQA & GQA** | 7.1 | shrinking the KV cache, the heads/quality tradeoff | reason about memory vs quality | ✅ (`NB28`) |
| 7.3 | **FlashAttention** | 7.1 | tiling, online softmax, HBM vs SRAM | explain hardware-aware attention | ✅ (`NB29`) |

### Part 8 — Scaling the Architecture
| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 8.1 | **Mixture of Experts (MoE)** | 7.x, 5.7 | sparse routing, top-k experts, load balancing | explain sparse scaling (Mixtral-style) | ✅ (`NB30`) |
| 8.2 | **Training on many GPUs** | 5.2, 5.7 | DDP, FSDP/ZeRO sharding, the params+grads+optimizer-state memory bill | read a frontier model's training-infrastructure section | 🔜 planned |

> **Why MoE now follows scaling laws (5.7):** MoE is an *answer* to the question
> scaling laws ask — how do you grow capacity ($N$) without paying full compute per
> token? Teaching the answer first left the motivation dangling.

### Part 9 — Production & Serving
*Goal: the systems that make serving an LLM viable — plus the graduation lap.*

| # | Module | Prereq | You will learn | After this you can | Status |
|---|--------|--------|----------------|--------------------|--------|
| 9.1 | **Quantization** | 7.x | absmax, zero-point, INT8, memory savings | shrink a model for deployment | ✅ (`NB31`) |
| 9.2 | **Speculative decoding** | 7.1 | draft+verify, accept/reject, the speedup | explain fast multi-token generation | ✅ (`NB32`) |
| 9.3 | **PagedAttention & continuous batching** | 7.1 | block tables, memory paging, batching | explain vLLM-style serving + capstone synthesis | ✅ (`NB33`) |
| 9.4 | **Graduation: reading a real LLM** | 9.3 | our library ↔ Llama-3 ↔ GPT-2 name map, counting Llama-3-8B's parameters by hand, what production code adds | open nanoGPT / Llama source and **recognize every line** | ✅ (`NB34`) |

> **Why 9.4 exists:** the point of building from scratch was never the tiny model —
> it was *transferable understanding*. This module cashes that in: you compute
> Llama-3-8B's 8.03B parameters from its config using only components you built,
> then walk a reading list of real model source files.

---

## Appendix A — Module → notebook file map

The notebook files are numbered to follow this curriculum order.

| Module | Topic | Notebook file |
| --- | --- | --- |
| 0.0 | Prologue | `docs/00_prologue.md` |
| 0.1 | Setup & PyTorch crash course | `notebooks/00_prerequisites/01_setup_and_pytorch.ipynb` |
| 0.2 | Neural networks in 30 min | `notebooks/00_prerequisites/02_neural_networks_in_30_min.ipynb` |
| 1.1 | The math you'll actually need | `notebooks/01_ml_foundations/03_math_primer.ipynb` |
| 1.2 | Tensors & linear algebra | `notebooks/01_ml_foundations/04_linear_algebra.ipynb` |
| 1.3 | Probability & calculus | `notebooks/01_ml_foundations/05_probability_and_calculus.ipynb` |
| 1.4 | Backprop from scratch | `notebooks/01_ml_foundations/06_backprop_from_scratch.ipynb` |
| 2.1 | Tokenization & embeddings | `notebooks/02_text_to_vectors/07_tokenization_and_embeddings.ipynb` |
| 2.2 | Positional encoding | `notebooks/02_text_to_vectors/08_positional_encoding.ipynb` |
| 3.1 | Self-attention | `notebooks/03_attention/09_self_attention.ipynb` |
| 3.2 | Multi-head attention & causal masking | `notebooks/03_attention/10_multihead_and_masking.ipynb` |
| 4.1 | Encoder layer | `notebooks/04_transformer/11_the_encoder_layer.ipynb` |
| 4.2 | Decoder layer | `notebooks/04_transformer/12_the_decoder_layer.ipynb` |
| 4.3 | Full transformer | `notebooks/04_transformer/13_full_transformer.ipynb` |
| 4.4 | Assemble your decoder-only GPT | `notebooks/04_transformer/14_assemble_gpt.ipynb` |
| 5.0 | From corpus to token stream | `notebooks/05_training/15_data_and_token_streams.ipynb` |
| 5.1 | Loss | `notebooks/05_training/16_cross_entropy_loss.ipynb` |
| 5.2 | Training loop | `notebooks/05_training/17_the_training_loop.ipynb` |
| 5.3 | Capstone: train your GPT | `notebooks/05_training/18_train_your_own_gpt.ipynb` |
| 5.4 | When training goes wrong | `notebooks/05_training/19_when_training_goes_wrong.ipynb` |
| 5.5 | Decoding & sampling | `notebooks/05_training/20_decoding_and_sampling.ipynb` |
| 5.6 | Evaluation | `notebooks/05_training/21_evaluating_a_language_model.ipynb` |
| 5.7 | Scaling laws | `notebooks/05_training/22_scaling_laws.ipynb` |
| 6.1 | SFT | `notebooks/06_finetuning/23_supervised_fine_tuning.ipynb` |
| 6.2 | LoRA | `notebooks/06_finetuning/24_peft_and_lora.ipynb` |
| 6.3 | Reward models & RLHF | `notebooks/06_finetuning/25_reward_models_and_rlhf.ipynb` |
| 6.4 | DPO | `notebooks/06_finetuning/26_dpo_preference_alignment.ipynb` |
| 7.1 | KV caching | `notebooks/07_inference/27_kv_caching.ipynb` |
| 7.2 | MQA/GQA | `notebooks/07_inference/28_advanced_attention.ipynb` |
| 7.3 | FlashAttention | `notebooks/07_inference/29_flash_attention.ipynb` |
| 8.1 | MoE | `notebooks/08_scaling/30_mixture_of_experts.ipynb` |
| 9.1 | Quantization | `notebooks/09_production/31_quantization_fundamentals.ipynb` |
| 9.2 | Speculative decoding | `notebooks/09_production/32_speculative_decoding.ipynb` |
| 9.3 | PagedAttention | `notebooks/09_production/33_paged_attention.ipynb` |
| 9.4 | Graduation: reading a real LLM | `notebooks/09_production/34_reading_a_real_llm.ipynb` |

## Appendix B — The applied track (Part II)

Application-builder topics live in **Part II — Applied LLM Engineering**
([CURRICULUM_PART_II.md](./CURRICULUM_PART_II.md), notebooks `35–41`):
prompt engineering (zero/few-shot, chain-of-thought, structured output) ·
embeddings for **semantic search & clustering** · **RAG** (chunking, retrieval,
hybrid search, reranking, context construction) · **evaluation** of generations
(hallucination, groundedness, faithfulness, relevance). Still future work there:
**agents** (tool calling, planning, memory, multi-agent) and **production apps**
(response caching, monitoring, observability, guardrails, fallbacks,
cost/latency budgeting).

Listed so they're recognized as a *separate scope decision*, not oversights.
