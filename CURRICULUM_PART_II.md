# 🧰 LLM Workout — Part II: Applied LLM Engineering (Proposed)

> **Status: PROPOSED OUTLINE — not yet built.** This is the blueprint for a
> second track, the way [CURRICULUM.md](./CURRICULUM.md) was the blueprint for
> Part I. Nothing here is written yet; this document is for agreeing on scope and
> sequence before any notebook is created.

## Why a Part II

Part I makes you a **model-builder**: you can build, train, align, and optimize an
LLM from scratch. But most "AI Engineer" work is **building systems that *use*
models** — prompting, retrieval, evaluation, agents, and production serving. Those
were deliberately out of scope in Part I (see CURRICULUM.md, Appendix B). Part II
is that missing half.

| | Part I (done) | Part II (this outline) |
| --- | --- | --- |
| You learn to | build the model | build products on top of models |
| Mindset | "how does it work inside?" | "how do I make it reliable, grounded, cheap, safe?" |

## Confirmed decisions

- **Build philosophy: from-scratch-first.** Build a tiny version of each idea with
  minimal dependencies (a brute-force vector index in NumPy, a hand-rolled RAG
  pipeline, eval metrics computed by hand), *then* point to the real production
  tool. Keeps the course's "understand by building" DNA.
- **Model access: local small model by default, optional hosted API.** Labs run
  free and reproducible on small *pretrained* models; an optional OpenAI-compatible
  path is shown (commented) for stronger output. (Part I's from-scratch char model
  can't follow instructions or embed arbitrary text, so Part II uses pretrained
  models as black boxes — the *systems* around them are what we build.)
- **First scope: the RAG spine (Sections A–E).** Using LLMs → Prompting →
  Embeddings/Search → RAG → Evaluation. Agents (F) and Production (G) come later.
- **Numbering:** continues the flat scheme — Part II notebooks are `30+` in
  `notebooks/`.

### Verified local stack (probed, works on CPU)

| Need | Choice | Notes |
| --- | --- | --- |
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` (~80 MB, 384-dim) | semantic search ranks correctly; index built from scratch in NumPy |
| Generation | `transformers` · `HuggingFaceTB/SmolLM2-135M-Instruct` (~270 MB) | loads on CPU, ~0.6 s/short answer; weak but coherent enough to show mechanics |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | small cross-encoder for Section D.4 |
| Optional API | `openai` (OpenAI-compatible) | better output; needs a key; never required to run a lab |

Installed via `pip install -e ".[applied]"`. Because these labs download models and
are slower, Part II notebooks run in a **separate CI lane** from the strict Part I
notebook-execution check.

### Build plan (Sections A–E consolidated into notebooks)

| Notebook | Covers | Needs |
| --- | --- | --- |
| `30_calling_a_model` | A.1 Calling a model + A.2 decoding in practice | local LLM |
| `31_structured_output` | A.3 JSON / schema output + validation + retries | local LLM |
| `32_prompt_engineering` | B.1 zero/few-shot · B.2 chain-of-thought · B.3 reliability | local LLM |
| `33_embeddings_and_semantic_search` | C.1 embeddings · C.2 from-scratch search · C.3 clustering | embeddings |
| `34_rag_retrieval` | D.1 why RAG · D.2 chunking · D.3 dense/keyword/hybrid retrieval | embeddings |
| `35_rag_generation` | D.4 reranking · D.5 context construction · D.6 end-to-end capstone | embeddings + LLM |
| `36_evaluating_llm_systems` | E.1–E.4 groundedness/faithfulness, LLM-as-judge, hallucination, regression | LLM |

---

## Proposed learning path

Legend: every module below is 🟢 **proposed / not yet written**. "Builds on" points
to Part I modules where relevant.

### Part A — Using LLMs in Practice
*Goal: drive a model as a black box and understand its knobs and costs.*

| # | Module | Prerequisites | You will learn | After this you can |
|---|--------|---------------|----------------|--------------------|
| A.1 | **Calling a model** | Part I (or none) | chat vs. completion, messages/roles, tokens, latency & cost basics, streaming | call a local or hosted LLM and reason about token cost |
| A.2 | **Decoding in practice** | Part I 5.4 (Decoding & Sampling) | applying temperature/top-p/stop-sequences via an API; determinism vs. creativity | tune generation for a task |
| A.3 | **Structured output** | A.1 | JSON output, schema-constrained decoding, parsing & validation, retries on bad output | get reliable machine-readable output |

### Part B — Prompt Engineering
*Goal: get reliable behavior from a fixed model by changing the input.*

| # | Module | Prereq | You will learn | After this you can |
|---|--------|--------|----------------|--------------------|
| B.1 | **Zero-shot & few-shot** | A.x | instructions, demonstrations, system prompts, prompt templates | design prompts that generalize |
| B.2 | **Chain-of-thought & reasoning** | B.1 | step-by-step prompting, self-consistency, when reasoning helps/hurts | improve multi-step task accuracy |
| B.3 | **Prompt reliability** | B.1 | output contracts, guard against prompt injection, versioning prompts | ship prompts you can trust |

### Part C — Embeddings & Semantic Search
*Goal: represent meaning as vectors and search by similarity.*

| # | Module | Prereq | You will learn | After this you can |
|---|--------|--------|----------------|--------------------|
| C.1 | **Embeddings for meaning** | Part I 2.1 (Tokenization & Embeddings) | sentence/document embeddings, cosine similarity, the geometry of meaning | turn text into searchable vectors |
| C.2 | **Semantic search** | C.1 | build a brute-force vector index, top-k nearest neighbors, then ANN intuition | build a "search by meaning" system |
| C.3 | **Clustering & exploration** | C.1 | k-means on embeddings, topic discovery, dimensionality reduction for viz | organize a corpus by meaning |

### Part D — Retrieval-Augmented Generation (RAG)
*Goal: ground a model in your own documents to cut hallucination. The centerpiece of Part II.*

| # | Module | Prereq | You will learn | After this you can |
|---|--------|--------|----------------|--------------------|
| D.1 | **Why RAG** | C.2 | parametric memory limits, hallucination examples, the retrieve→augment→generate loop | explain when/why to use RAG vs. fine-tuning |
| D.2 | **Chunking** | D.1 | splitting strategies (size, overlap, semantic), metadata | prepare documents for retrieval |
| D.3 | **Retrieval** | D.2, C.2 | dense vs. keyword (BM25) vs. **hybrid** search | retrieve relevant context reliably |
| D.4 | **Reranking** | D.3 | cross-encoder reranking, why first-stage recall ≠ final precision | sharpen retrieved results |
| D.5 | **Context construction** | D.4 | prompt assembly, context-window budgeting, citations, lost-in-the-middle | feed a model grounded context well |
| D.6 | **RAG capstone** | D.1–D.5 | build an end-to-end RAG Q&A system over a small corpus | ship a working RAG app |

### Part E — Evaluating LLM Systems
*Goal: measure generation quality — the thing Part I's perplexity can't.*

| # | Module | Prereq | You will learn | After this you can |
|---|--------|--------|----------------|--------------------|
| E.1 | **Why eval is hard** | Part I 5.5 (Evaluation) | open-ended output, no single ground truth, beyond perplexity | frame an eval for a real task |
| E.2 | **RAG/answer metrics** | D.6 | groundedness, faithfulness, relevance, answer correctness | quantify a RAG system's quality |
| E.3 | **LLM-as-judge** | E.1 | using a model to grade outputs, rubric design, bias/limits | build scalable automated eval |
| E.4 | **Hallucination detection & regression testing** | E.2 | detecting unsupported claims, building an eval set, catching regressions in CI | keep quality from silently degrading |

### Part F — Agents
*Goal: let a model take actions, plan, and use tools.*

| # | Module | Prereq | You will learn | After this you can |
|---|--------|--------|----------------|--------------------|
| F.1 | **Tool / function calling** | A.3 | exposing tools, the call→execute→observe loop, schemas | give a model real capabilities |
| F.2 | **Planning loops (ReAct)** | F.1 | reason-act-observe, multi-step task decomposition, stopping criteria | build an agent that solves multi-step tasks |
| F.3 | **Memory** | F.2 | short-term (scratchpad) vs. long-term (vector memory), state | give agents continuity |
| F.4 | **Multi-agent systems** | F.2 | roles, orchestration, hand-offs, failure modes | coordinate multiple agents |

### Part G — Production Engineering
*Goal: make an LLM app fast, cheap, observable, and safe.*

| # | Module | Prereq | You will learn | After this you can |
|---|--------|--------|----------------|--------------------|
| G.1 | **Latency & cost** | A.1 | streaming, batching, model selection, token budgeting | hit latency/cost targets |
| G.2 | **Caching** | G.1 | exact & **semantic** caching, prompt caching, invalidation | cut cost and latency |
| G.3 | **Observability** | F.x | tracing requests, logging prompts/outputs, monitoring quality & drift | debug and watch a live system |
| G.4 | **Guardrails & fallbacks** | B.3, E.4 | input/output validation, safety filters, retries, graceful degradation | ship something safe and resilient |

---

## Capstone for Part II

A small but real application that stacks the track: a **grounded, evaluated,
tool-using assistant** over a chosen corpus — RAG for knowledge (Part D), an eval
suite gating changes (Part E), a tool/agent step (Part F), and caching +
guardrails + tracing (Part G). Ideally it can use the model the student trained in
Part I as the (small, local) generator, closing the loop across both tracks.

## Status

Build decisions are confirmed (see above) and the local stack is verified.
Building Sections **A–E** as notebooks `27–33`. Sections **F (Agents)** and
**G (Production)** remain outlined-but-unbuilt, to follow once the spine lands.
