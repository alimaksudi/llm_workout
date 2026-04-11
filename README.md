# 🚀 LLM Workout: From Zero to Transformer Master

A hands-on journey to mastering Transformer architectures, LLMs, and the mathematics that power them. This repository is designed to bridge the gap between academic papers and production-ready implementation.

## 📌 Project Goals
- **Architecture First**: We focus on building the inner workings of Transformers from scratch.
- **Visual Learning**: Every component is accompanied by a Jupyter Notebook explaining the "Why" behind the tensors.
- **Modern Standards**: We don't just stop at the 2017 paper; we implement Llama-style improvements (RoPE, RMSNorm, SwiGLU).

## 📚 Architectural Textbook (Zero to Hero)
For a deep dive into the mathematical concepts, analogies, and strictly non-code architectural theory driving these Jupyter notebooks, refer to our compiled documentation:
- [Chapter 1: Mathematics and Structural Building Blocks](./docs/01_math_and_building_blocks.md)
- [Chapter 2: Architecture Assembly](./docs/02_architecture_assembly.md)
- [Chapter 3: Efficiency and Scaling](./docs/03_efficiency_and_scaling.md)
- [Chapter 4: Training and Alignment](./docs/04_training_and_alignment.md)
- [Chapter 5: Advanced Hardware & Mixture of Experts (MoE)](./docs/05_advanced_hardware_and_moe.md)
- [Chapter 6: Production and Inference Engineering](./docs/06_production_and_inference.md)

## 🗺️ Roadmap & Progress Tracker

### Module 1: Math Foundations (CORE)
- [x] **01. Tensors & Linear Algebra**: Matrix operations, dimensionality tracking, and dot products.
- [x] **02. Probability & Calculus**: Understanding Softmax, Cross-Entropy, and Gradients.

### Module 2: Building Blocks
- [x] **03. Tokenization & Embeddings**: Turning text into numbers (BPE, WordPiece).
- [x] **04. Positional Encoding**: Injecting sequence order (Sin/Cos vs. RoPE).
- [x] **05. Attention Mechanisms**: Scaled Dot-Product, Self-Attention, and Multi-Head Attention.

### Module 3: Architecture Assembly
- [x] **06. The Encoder Layer**: Residual connections, LayerNorm, and FFN.
- [x] **07. The Decoder Layer**: Masked Attention and Cross-Encoder Attention.
- [x] **08. Full Transformer**: Putting it all together.

### Module 4: Efficiency & Scaling
- [x] **09. KV Caching**: Optimizing inference speed.
- [x] **10. Advanced Attention**: GQA (Grouped Query) and MQA (Multi-Query).

### Module 5: Pre-Training Mechanics
- [x] **11. Next-Token Prediction & Loss**: Cross-Entropy Loss logic.
- [x] **12. The Training Loop**: Gradients, Optimizers (AdamW), and Weight Decay.

### Module 6: Fine-Tuning & Alignment
- [x] **13. Supervised Fine-Tuning (SFT)**: Transitioning to Chatbot format.
- [x] **14. DPO Preference Alignment**: Mathematical formulation of DPO.

### Module 7: Applied LLM Engineering & PEFT
- [x] **15. LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning mathematics.

### Module 8: Advanced Architecture
- [x] **16. Mixture of Experts (MoE)**: Routing mathematics for sparse scaling.

### Module 9: Hardware Optimization
- [x] **17. FlashAttention**: Memory tiling and hardware-aware scaling.

### Module 10: Production & Inference Engineering
- [x] **18. Quantization Fundamentals**: Absmax, Zero-Point, and memory savings.
- [x] **19. Speculative Decoding**: Fast inference with Draft vs. Target models.
- [x] **20. Continuous Batching**: PagedAttention and KV Cache blocks (vLLM style).

## 🛠️ Setup
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies and the local `llm_workout` library
pip install -e .
```

## 🚀 Training Demo

We have extracted the core neural network layers from the educational notebooks into a modular, production-ready `src/llm_workout/` python library.

You can import these components directly into your own projects! The architecture uses modernized Llama-3 standards (RMSNorm, SwiGLU):
```python
from llm_workout.model import GPT
from llm_workout.layers import RMSNorm, SwiGLU

# Initialize a custom GPT
model = GPT(vocab_size=50000, d_model=256, num_layers=4, num_heads=8, hidden_dim=1024)
```

You can also train a miniature Transformer from scratch right in your terminal using the TinyShakespeare dataset! This standalone script downloads the data and streams its learning process to your console:

```bash
python scripts/train_tiny.py
```


---
*Created with ❤️ for the LLM community.*
