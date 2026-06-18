# Chapter 3: Efficiency and Scaling

The core architecture is correct, but it is inefficient in practice when generating tokens one at a time. This chapter breaks down how we optimize the inference pipeline so the model can generate text quickly.

## 1. The Autoregressive Bottleneck

In a Decoder-only model, generation happens one token at a time (autoregressively).
When generating the 100th word, the model takes words $1 \dots 99$ and calculates all their attention interactions to output word 100.
Then, to generate word 101, the naive implementation takes words $1 \dots 100$ and calculates all interactions *all over again* from scratch!

How expensive is this? At each step, attention compares every token against every other token, so a single step over a sequence of length $N$ costs about $\mathcal{O}(N^2)$ work. Because the naive loop redoes that full computation at every new token, generation slows down noticeably as the context window grows.

## 2. KV Caching

### Avoiding Recalculation
The past words (e.g., words 1-99) never change their meaning while we generate word 100, so their **Key (K)** and **Value (V)** vectors stay the same from one step to the next.

**KV Caching** uses this fact:
Instead of throwing away the computed $K$ and $V$ matrices after predicting a token, we save them in GPU memory.
When calculating word 101, we only run the calculations to find the $Q, K, V$ arrays for *just* the new 100th word, then append its $K$ and $V$ onto the end of our saved cache.

> [!NOTE]
> With the cache in place, each new token only computes one fresh Query and compares it against the $N$ cached Keys. That makes the work per generated token roughly $\mathcal{O}(N)$ instead of $\mathcal{O}(N^2)$ — a large saving, though each token still gets a little slower as the cache grows.

**The Exam Analogy**: Instead of rereading the entire 10-page reading comprehension article every time a new question is asked, you simply refer back to the highlighted bullet points (the KV Cache) you wrote down previously!

## 3. MQA and GQA (Scaling the Cache)

KV Caching dramatically reduces redundant compute, but it creates a new problem: **memory pressure.**
If 10,000 users are chatting with the model concurrently, the server has to store a separate KV cache for all 10,000 conversations at the same time in GPU VRAM.

How big is a cache? It scales roughly as:

$$\text{cache size} \approx 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{batch}$$

The factor of 2 is for K and V. The key lever we can pull is **`kv_heads`** — the number of distinct Key/Value sets we store. Shrink that, and the whole cache shrinks with it.

### Multi-Query Attention (MQA)
Standard Multi-Head Attention computes a *unique* set of Keys and Values for every Head (e.g., 32 Queries, 32 Keys, 32 Values). That makes the cache large.

MQA shrinks the cache by forcing all 32 Query heads to share *exactly $1$* set of Keys and Values:
- Cache size shrinks by about 32× (1 KV head instead of 32).
- Reasoning quality can drop a little, since all heads now read from the same shared associative memory.

### Grouped-Query Attention (GQA)
GQA (used by Llama 2's larger models and all of Llama 3) sits between full Multi-Head Attention and MQA.
Instead of 1 shared KV (MQA) or 32 individual KVs (standard MHA), it groups the query heads into a handful of clusters that each share a KV set.
- 32 Query Heads.
- 8 KV Heads.
- (Every 4 Queries share one group's KV cache.)

Here the mechanism does the talking: 8 KV heads instead of 32 makes the cache about **4× smaller** (compared to MQA's 32× reduction). So GQA recovers most of MQA's memory savings while keeping quality close to full MHA, because each cluster of heads still keeps its own Keys and Values rather than collapsing to a single shared set.

> [!NOTE]
> Exact head counts for closed models like GPT-4 are not public. The "32 heads" figures above are illustrative of a model in the Llama 2 7B size class — use them to reason about the *mechanism*, not as exact specs for any specific proprietary model.
