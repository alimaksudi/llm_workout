# 📖 Glossary — every term in plain English

Hit a word you don't know? Look it up here. **No jargon is used in this course
without being explained somewhere** — this page is the safety net, so you never need
to go searching the internet mid-lesson.

Terms are grouped by when you'll meet them. The module in *(brackets)* is where the
idea is taught properly.

> Math **symbols** (Σ, ∇, log, sin, Greek letters…) live in their own place:
> **Module 1.1 — The Math You'll Actually Need**
> (`notebooks/01_ml_foundations/03_math_primer.ipynb`), which teaches every one from zero.

---

## The basics

**Tensor** *(0.1)* — a container of numbers. A single number is 0-D, a list is 1-D
(a vector), a grid is 2-D (a matrix), and a stack of grids is 3-D. Everything in deep
learning is a tensor.

**Shape** *(0.1)* — how big a tensor is along each direction, e.g. `(2, 5, 128)`.
Reading shapes is the #1 debugging skill in this course.

**Parameter / weight** *(0.2)* — one of the adjustable numbers inside a model. "A 7B
model" means 7 billion of them. Training = nudging these numbers until the model is good.

**Bias** *(0.2)* — a parameter that is simply *added* (a baseline nudge), rather than
multiplied by an input.

**Activation function** *(0.2)* — a simple bend (like ReLU) applied after a layer.
Without it, stacking layers would collapse into one straight line and the network
couldn't model curves.

**Forward pass** *(0.2)* — running data *through* the model to get a prediction.

**Backward pass / backpropagation** *(1.4)* — walking back through the model to work
out how each parameter should change. This is what `loss.backward()` does.

**Gradient** *(1.1, 1.4)* — the slope: "if I nudge this parameter, how much does the
loss move, and in which direction?"

**Loss** *(0.2, 5.1)* — one number measuring how wrong the model is. Training pushes
it down.

**Optimizer** *(1.4, 5.2)* — the rule for applying gradients to parameters. SGD is the
simple one; **Adam / AdamW** are the adaptive ones every LLM uses.

**Learning rate** *(0.2, 5.2)* — the step size. Too small = training crawls; too big =
it explodes.

**Hyperparameter** *(4.3)* — a setting *you* choose before training (learning rate,
number of layers, batch size), as opposed to a **parameter**, which the model *learns*.

**Epoch** *(5.2)* — one full pass over the training dataset. (LLMs on huge corpora
often train for less than a single epoch — they see most text only once.)

**Batch** *(0.1, 5.3)* — a group of examples processed together in one step, for speed
and a less noisy gradient.

**Overfitting** *(0.2, 5.6)* — when a model memorizes the training data instead of
learning the pattern, so it does well on what it has seen and badly on anything new.

**Regularization** *(5.2)* — any technique that discourages memorization, to fight
overfitting. Weight decay (the "W" in AdamW) is one.

**Train / validation split** *(0.2, 5.6)* — holding back some data the model never
trains on, so you can honestly test whether it *generalized*.

---

## Text and models

**Token** *(2.1)* — the chunk of text a model actually reads. Usually a sub-word
piece, not a whole word: `"unbelievable"` might be `un | believ | able`.

**Tokenizer / BPE** *(2.1)* — the tool that splits text into tokens and maps them to
integer ids. **BPE** (Byte-Pair Encoding) is the algorithm we build from scratch.

**Vocabulary (vocab size)** *(2.1)* — the full set of tokens a model knows; its size is
how many distinct tokens exist (often 50k–130k).

**Embedding** *(2.1)* — the vector of numbers representing a token's meaning. The
embedding table is a big lookup: one row per token.

**Context window / sequence length** *(3.2)* — how many tokens the model can look at
at once.

**Logits** *(1.3, 5.1)* — the raw, unnormalized scores the model outputs — one per
vocabulary token — *before* softmax turns them into probabilities.

**Softmax** *(1.2)* — turns a list of scores into probabilities that are all positive
and sum to 1.

**Autoregressive** *(4.2)* — generating one token at a time, each new token
conditioned on everything written so far. This is how all GPT-style models write.

**Causal mask** *(3.1, 4.2)* — the block that stops a position from seeing tokens that
come *after* it, so the model can't cheat by peeking at the answer.

**Residual connection** *(4.1)* — adding a layer's input back to its output
(`x + layer(x)`), giving gradients a clean "highway" through deep networks.

**Normalization (LayerNorm / RMSNorm)** *(4.1)* — rescaling a vector so its numbers
stay in a sane range, which keeps deep networks trainable.

**Base model vs. instruction-tuned** *(6.1)* — a **base** model just continues text; an
**instruction-tuned** (chat) model has been fine-tuned to follow requests and answer.

---

## Training, tuning, and serving

**Pre-training** *(Part 5)* — the big, expensive run that teaches a model language from
a huge corpus.

**Fine-tuning** *(Part 6)* — further training of an already-trained model on a smaller,
specific dataset.

**SFT (Supervised Fine-Tuning)** *(6.1)* — fine-tuning on example conversations to
teach a model to chat.

**LoRA / PEFT** *(6.2)* — fine-tuning by training a small number of *added* parameters
while the original weights stay frozen. Cheap enough to run on consumer hardware.

**Alignment / RLHF / DPO** *(6.3, 6.4)* — training a model to match human *preferences*
(helpful, harmless), not just to imitate text.

**Inference** *(Part 7)* — *using* a trained model to produce output, as opposed to
**training** it. Inference has no backward pass, so it's much cheaper per token — but
you do it constantly in production, so speed matters enormously.

**Decoding / sampling** *(5.5)* — the rule for picking the next token from the model's
probabilities: greedy, temperature, top-k, top-p.

**Perplexity** *(5.6)* — `exp(loss)`; roughly "how many options is the model torn
between per token?" Lower is better.

**KV cache** *(7.1)* — storing the Keys and Values of past tokens so generation doesn't
recompute them every step. The single biggest generation speed-up.

**Quantization** *(9.1)* — storing weights in fewer bits (e.g. int8 instead of
float32) to shrink memory, at a small cost in precision.

**VRAM** — the memory on a GPU. The usual limiting factor: a model must fit in VRAM to
run fast.

**FLOPs** *(8.2)* — "floating-point operations": a unit of raw compute. Used to measure
how expensive training a model is.

**Latency vs. throughput** *(9.3)* — **latency** is how long *one* request takes;
**throughput** is how many requests you serve *per second*. Serving systems trade one
against the other.

**Checkpoint** *(5.3)* — a saved snapshot of a model's weights, so you can reload it
later without retraining.

---

## Applied (Part II)

**RAG (Retrieval-Augmented Generation)** *(D)* — retrieving relevant documents and
putting them in the prompt, so the model answers from *provided* text instead of
memory.

**Chunking** *(D)* — splitting documents into pieces small enough to retrieve and fit
in a prompt.

**Vector index / semantic search** *(C)* — finding text by *meaning* (comparing
embedding vectors) rather than by exact keywords.

**Hallucination** *(D, E)* — when a model states something fluent and confident that is
simply false.

**Prompt engineering** *(B)* — changing the *input* to get better behavior from a fixed
model (few-shot examples, chain-of-thought, output contracts).
