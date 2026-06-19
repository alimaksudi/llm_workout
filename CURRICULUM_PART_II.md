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

## A design decision to settle first

Part I is **from-scratch and dependency-light** (essentially just PyTorch). Part II
inherently touches the outside world — model APIs, vector indexes, eval harnesses,
serving infra. Two ways to teach it:

- **A. From-scratch-first (recommended).** Build a tiny version of each idea with
  minimal dependencies (e.g., a brute-force vector index in NumPy, a hand-rolled
  ReAct loop), *then* point to the real tool people use in production. Keeps the
  course's "understand by building" DNA and avoids framework churn.
- **B. Tool-first.** Teach directly with the popular stack (a hosted LLM API, a
  vector DB, an agent framework). Faster to "real," but dates quickly and hides
  the mechanics.

This outline assumes **Option A**. A further sub-decision: whether hands-on labs
may call a **hosted LLM API** (needs a key, costs cents) or must run a **local
small model** (free, reproducible, weaker). Recommendation: make labs work with a
local model by default and offer an optional hosted-API path.

> ⚠️ These two choices change a lot of the content. They should be confirmed
> before building.

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

## Open questions to confirm before building

1. **Build philosophy:** Option A (from-scratch-first) or B (tool-first)?
2. **Model access:** local small model by default, hosted API, or both?
3. **Scope/length:** all of Parts A–G, or a focused subset first (e.g., A–E, the
   highest-value "prompt → search → RAG → eval" spine)?
4. **Numbering:** continue the flat notebook numbering (27+) or start Part II in a
   separate `notebooks/part2/` directory?
