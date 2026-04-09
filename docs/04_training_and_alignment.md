# Chapter 4: Training and Alignment

The architecture is wired and numerically optimized, but right now the internal tensors are just filled with randomized floating-point noise. This chapter intimately covers the sequential phases of gradient descent used to literally mathematically teach the numbers how to speak and reason in English.

## Phase 1: Pre-Training (The Foundation Model)

**Objective**: Read the entire internet corpus and learn the deep statistical physics underpinning human language logic.

### Cross-Entropy Loss
How does the model quantitatively know if it guessed the right word? 
The model natively outputs an un-normalized "Logit" array, which Softmax converts to a probability distribution (e.g., 90% chance the next word is `dog`, 10% chance it is `cat`). 
The true dataset reality (the textbook it is reading) is physically parsed as a "One-Hot Vector" (100% `dog`, 0% for every other word).

**Cross-Entropy Loss** calculates the direct mathematical distance vector between what the model guessed conceptually, and reality. If the numerical distance (Loss scalar) is high, the model performed poorly!

### The Optimizer (AdamW) & Backpropagation
Once we calculate the exact Loss scalar, we execute **Backpropagation** (applying the Calculus Chain Rule backwards through the network layers) to explicitly deduce exactly which individual matrix weights in the network were mathematically responsible for producing the bad guess.
The **AdamW Optimizer** ingests these error gradients to explicitly nudge the numeric weights slightly closer to reality.

**The Analogy**: Imagine adjusting thousands of radio dials. The Loss tells you there is static on the channel. Backpropagation tells you definitively to turn Dial A to the right and Dial B to the left.

*Result*: Out comes a "Base Model" (Like Llama 3 Base). It does not answer questions; it only predicts the next logical sequence word (e.g., Prompt: "What is the capital of France?", Output: "What is the capital of Spain?").

## Phase 2: Supervised Fine-Tuning (SFT)

**Objective**: Teach the base model to organically halt its endless rambling and strictly behave like a contained, helpful Assistant.

We transition datasets from raw Wikipedia webs-scrapes to strict Question & Answer dialog formatting.
We structurally wrap the dataset strings in specialized formatting tokens (like `<|im_start|>user\\n Hello <|im_end|>\\n<|im_start|>assistant\\n Hi!`).

### Loss Masking
During SFT, we emphatically do *not* want the model to natively learn how to predict the User's prompts! We exclusively want it to learn how to predict the Assistant's correct replies.
We mechanically apply a binary boolean `Mask` to the Cross-Entropy Loss step, physically zeroing out the loss scalar calculations overriding the User's textual tokens. The model matrix is only academically graded on updating logic related to its actual conversational responses!

## Phase 3: Alignment (DPO/RLHF)

**Objective**: Teach the Chatbot matrix to be subjectively pleasant, physically safe, and aligned with complex human preferences.

Standard SFT mechanics organically create a bot that technically answers questions, but it might answer them rudely, incorrectly, or dangerously.
- **Old Methodology (RLHF)**: Spin up an entirely conceptually separate "Reward Model" network that continuously monitors and judges the core Chatbot's responses and hands out scalar numerical points. Extremely brittle to code and computationally massively expensive.
- **Modern Methodology (DPO - Direct Preference Optimization)**: By fundamentally rethinking the mathematical loss formulas, DPO structurally completely eliminates the Reward Model! We pass in a curated dataset containing a dynamically selected *Chosen* (good) answer and a *Rejected* (bad) answer. The DPO loss equation explicitly pushes the target probability of the Chosen answer algebraically UP, while simultaneously pulling the probability of the Rejected answer violently DOWN organically inside the generation matrix layer!
