# Chapter 5: Advanced Hardware & Mixture of Experts (MoE)

When you scale a Transformer up to 70 billion or even a trillion parameters, you start hitting real hardware limits: how much memory a GPU has, and how fast it can move data around. This chapter covers three of the engineering tricks that make modern large language models practical: LoRA, Mixture of Experts, and FlashAttention.

## 1. Parameter-Efficient Fine-Tuning (PEFT) & LoRA

Suppose you want to fine-tune a 70B parameter model. The problem isn't just storing the weights, it's everything training adds on top of them.

When you train with an optimizer like AdamW, memory is needed for several copies of the model:

- **The weights themselves** (1x).
- **The gradients**, one number per weight (1x).
- **Adam's first moment `m`** (the running average of gradients), one per weight (1x).
- **Adam's second moment `v`** (the running average of squared gradients), one per weight (1x).

So full fine-tuning needs roughly **4x** the memory of the model just to hold weights, gradients, and optimizer state, before counting activations. For a large model that runs into hundreds of gigabytes of VRAM.

### LoRA (Low-Rank Adaptation)

LoRA sidesteps most of that cost:

1. It **freezes** the original weight matrices. Because they never change, no gradients or optimizer state are needed for them.
2. It adds two small "low-rank" matrices alongside each frozen matrix.
3. During backpropagation, only those small matrices get updated.

Since only the small matrices are trained, the expensive gradient and optimizer memory is needed only for them, which is a tiny fraction of the full model.

**The dictionary analogy**: You don't rewrite a 5,000-page encyclopedia just to learn a few new slang words. You jot the new words on a few sticky notes and stick them to the back cover. The encyclopedia stays as it is.

## 2. Sparse Mixture of Experts (MoE)

To make a model "smarter," the usual move is to make its feed-forward layers wider and deeper. But a bigger dense model is also slower to run, because every token passes through every parameter.

**How do we add parameters without paying for all of them on every token?**

**MoE** splits one large feed-forward layer into several smaller feed-forward networks called "experts." Instead of every token going through one giant matrix, a small **router** network scores the experts for each token and sends the token to only the top few (often the top 2).

```mermaid
graph TD
    Token[Input Token Dim: 4096] --> Router[Softmax Router Network]

    Router -->|Top 1: Math Score| E1[Expert 1<br>Mathematics Feed-Forward]
    Router -.->|Not Chosen| E2[Expert 2]
    Router -.->|Not Chosen| E3[Expert 3]
    Router -->|Top 2: Code Score| E4[Expert 4<br>Python Feed-Forward]

    E1 -->|Multiply by Router Confidence| Sum{+}
    E4 -->|Multiply by Router Confidence| Sum

    Sum --> Out[Output Prediction]
```

**The construction analogy**:
- **Dense network**: One "jack-of-all-trades" worker handles every task. Whatever comes in, this single worker processes it. Simple, but slow as the work grows.
- **MoE**: You hire 8 specialists (the experts) plus 1 foreman (the router). When a task arrives, the foreman sends it to the right specialist. The other 7 stay idle for that task, so you do less work per token.

For example, *Mixtral 8x7B* holds roughly 47 billion parameters in memory, but because only 2 of its 8 experts run per token, each token passes through about 13 billion parameters. The result is roughly the quality of a much larger dense model at the inference speed of a far smaller one.

A catch worth knowing: if left alone, the router tends to favor a few experts and ignore the rest. Training therefore adds an auxiliary **load-balancing loss** that penalizes uneven routing, so all experts get used.

## 3. FlashAttention

The last wall is memory bandwidth. A GPU has two kinds of memory that matter here:

1. **HBM (High Bandwidth Memory)**: Large (tens of gigabytes) but comparatively slow to read and write.
2. **SRAM (on-chip memory)**: Tiny (megabytes) but very fast.

Standard attention computes $Q \cdot K^T$, writes the full $N \times N$ score matrix to HBM, reads it back for the softmax, writes it again, then reads it once more for the multiply by $V$. A large fraction of the time goes to moving this matrix in and out of HBM rather than to arithmetic. (Think of repeatedly walking to a slow fridge instead of cooking, an illustrative way to picture the bottleneck.)

### Memory tiling

FlashAttention restructures the computation to avoid those round trips:

It splits $Q$, $K$, and $V$ into small **tiles** that fit in fast SRAM. Using an "online softmax" that updates a running maximum and a running denominator as it goes, it computes the output one tile at a time, on-chip, without ever writing the full $N \times N$ attention matrix to HBM.

By removing those $O(N^2)$ reads and writes to HBM, FlashAttention makes long sequences far more practical. It is still bandwidth-sensitive, but it moves much less data. This is a big part of why context windows have grown from a few thousand tokens toward a million or more.

***

*That covers three of the core tricks behind modern large models: LoRA for cheap fine-tuning, MoE for scaling parameters without scaling compute per token, and FlashAttention for fitting attention into the GPU's fast memory. You now have a solid conceptual map of how today's systems are made to train and run efficiently.*
