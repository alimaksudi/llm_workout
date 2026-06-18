# Chapter 4: Training and Alignment

The architecture is wired up, but right now the internal tensors are just filled with random floating-point noise. This chapter covers the sequential phases of gradient descent used to teach those numbers how to produce fluent, helpful language.

## Phase 1: Pre-Training (The Foundation Model)

**Objective**: Read a huge corpus of text and learn the statistical patterns underlying human language.

### Cross-Entropy Loss
How does the model know if it guessed the right word?
The model outputs an un-normalized "Logit" array, which Softmax converts to a probability distribution (e.g., 90% chance the next word is `dog`, 10% chance it is `cat`).
The true target from the dataset is encoded as a "One-Hot Vector" (100% `dog`, 0% for every other word).

**Cross-Entropy Loss** turns this into a single number: it is *high* when the model gave low probability to the correct word, and *near zero* when the model was confident and correct. It is a scalar score, not a distance vector — there is just one number per prediction telling us how surprised the model was.

A tiny worked example. The loss for one prediction is $-\log(p_{\text{correct}})$:
- If the model gave the correct word a probability of $0.9$: loss $= -\log(0.9) \approx 0.11$ (small — good guess).
- If the model gave the correct word a probability of $0.1$: loss $= -\log(0.1) \approx 2.30$ (large — bad guess).

### The Optimizer (AdamW) & Backpropagation
Once we have the Loss scalar, we run **Backpropagation** (the Calculus Chain Rule applied backwards through the network) to figure out which individual weights were responsible for the bad guess.
The **AdamW Optimizer** takes these gradients and nudges the weights slightly in the direction that lowers the loss.

**The Analogy**: Imagine adjusting thousands of radio dials. The Loss tells you there is static on the channel. Backpropagation tells you which direction to turn each dial to reduce the static.

*Result*: Out comes a "Base Model" (like Llama 3 Base). It does not *answer* questions — it only *continues* text. Given "What is the capital of France?", a base model is just as likely to continue with more questions ("What is the capital of Spain? What is the capital of...") as it is to answer, because all it learned to do is predict plausible next text, not respond to a request. (We'll see this same "continuation, not answering" behavior in the training loop notebook, where "Hello" continues into "Hello, World!".)

## Phase 2: Supervised Fine-Tuning (SFT)

**Objective**: Teach the base model to stop rambling and behave like a helpful Assistant.

We switch datasets from raw web text to Question & Answer dialog formatting.
We wrap the data in special formatting tokens (like `<|im_start|>user\n Hello <|im_end|>\n<|im_start|>assistant\n Hi!`).

### Loss Masking
During SFT, we do *not* want the model to learn to predict the User's prompts — we only want it to learn to predict the Assistant's replies.
The mechanism is simple: for every token that belongs to the user/prompt portion, we set its target label to the special `ignore_index` value of `-100`. PyTorch's cross-entropy function skips any target equal to `ignore_index`, so those tokens contribute zero loss. Only the assistant's tokens produce gradients, so the model is graded only on its own responses. (This is the same Cross-Entropy and label-shifting machinery from Notebooks 11 and 13, just with some target labels blanked out.)

## Phase 3: Alignment (DPO/RLHF)

**Objective**: Make the assistant pleasant, safe, and aligned with human preferences.

SFT produces a bot that answers questions, but it might answer rudely, unhelpfully, or unsafely.
- **RLHF (Reinforcement Learning from Human Feedback)**: Train a separate "Reward Model" network that scores the chatbot's responses, then use reinforcement learning (PPO) to push the chatbot toward higher-scoring answers. RLHF was the breakthrough that made models like ChatGPT possible and is still widely used in practice. Its main downsides are complexity and cost: you train and run an extra reward model and a full RL loop.
- **DPO (Direct Preference Optimization)**: DPO rewrites the math so you no longer need a *separately trained* reward model. You feed in pairs of a *Chosen* (good) answer and a *Rejected* (bad) answer, and the DPO loss directly raises the probability of the chosen answer and lowers the probability of the rejected one. DPO is simpler and cheaper than RLHF, but note it does *not* eliminate every extra model: the loss still compares against a frozen **reference model** (typically the SFT model) to keep the tuned model from drifting too far from its original behavior.
