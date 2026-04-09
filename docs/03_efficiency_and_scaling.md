# Chapter 3: Efficiency and Scaling

The core architecture is computationally flawless, but terribly inefficient in practice when generating tokens sequentially. This chapter breaks down how we mathematically optimize the inference pipeline so the model can generate text rapidly.

## 1. The Autoregressive Bottleneck

In a Decoder-only model, generation happens one single token at a time (autoregressively).
When generating the 100th word, the massive model matrix takes words $1 \\dots 99$ and calculates all their attention interactions to output word 100.
Then, to generate word 101, the naive implementation takes words $1 \\dots 100$ and calculates all interactions *all over again* from scratch!

This quadratic $\\mathcal{O}(N^2)$ scalar recalculation causes generation speeds to aggressively slow down the longer the context window gets.

## 2. KV Caching

### Avoiding Recalculation
Since the past words (e.g., words 1-99) NEVER alter their meaning while we generate word 100, their purely mathematically computed **Key (K)** and **Value (V)** embedding vectors remain entirely static.

**KV Caching** physically solves the bottleneck:
Instead of lazily throwing away the computed $K$ and $V$ matrices after predicting a token, we physically save them incrementally into the GPU's memory block.
When calculating word 101, we only run the heavy calculations to find the $Q, K, V$ arrays for *just* the new 100th word, and trivially append its $K$ and $V$ onto the end of our saved memory list.

> [!NOTE]
> The model only ever fully calculates Attention scores for the *newest token*, mathematically comparing its single Query vector against the massive cache of historical Keys.

**The Exam Analogy**: Instead of explicitly returning to reread the entire 10-page reading comprehension article every time a new test question is asked, you simply refer back to the exact highlighted bullet points (The KV Cache) you wrote down previously!

## 3. MQA and GQA (Scaling the Cache)

While KV Caching perfectly fixes compute delays, it creates a massive new physical problem: **Memory Explosions.**
If 10,000 users are chatting with the model concurrently, the server has to physically store massive numerical KV caches for all 10,000 unique conversations simultaneously in GPU VRAM!

### Multi-Query Attention (MQA)
Standard Multi-Head Attention mechanically calculates a *unique* set of Keys and Values for every single computation Head (e.g., 32 Queries, 32 Keys, 32 Values). This means the Cache size is massive.

MQA aggressively shrinks the physical cache by forcing all 32 Query heads to share *exactly $1$ set* of Keys and Values!
- Nuanced reasoning quality drops slightly (since all heads share the exact same associative memory base).
- Physical Cache size shrinks dramatically by 32x!

### Grouped-Query Attention (GQA)
GQA (native to modern models like Llama 3) finds the perfect golden mathematical ratio stabilizing the gap between Standard Attention and MQA.
Instead of 1 Shared KV (MQA) or 32 Individual KVs (Standard), it organically groups the heads into clusters.
- 32 Query Heads.
- 8 KV Heads.
(Every 4 Queries share a single group's KV cache).

This physical grouping structurally provides 80% of the VRAM memory savings of MQA, while simultaneously retaining 99% of the reasoning quality of Full standard Attention!
