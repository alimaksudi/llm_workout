# 🚀 LLM Workout: From Zero to Transformer Master

A hands-on journey to mastering Transformer architectures, LLMs, and the mathematics that power them. This repository is designed to bridge the gap between academic papers and production-ready implementation.

## 📌 Project Goals
- **Architecture First**: We focus on building the inner workings of Transformers from scratch.
- **Visual Learning**: Every component is accompanied by a Jupyter Notebook explaining the "Why" behind the tensors.
- **Modern Standards**: We don't just stop at the 2017 paper; we implement Llama-style improvements (RoPE, RMSNorm, SwiGLU).

## 🗺️ Roadmap & Syllabus

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
- [ ] **11. Next-Token Prediction & Loss**: Cross-Entropy Loss logic.
- [ ] **12. The Training Loop**: Gradients, Optimizers (AdamW), and Weight Decay.

### Module 6: Fine-Tuning & Alignment
- [x] **13. Supervised Fine-Tuning (SFT)**: Transitioning to Chatbot format.
- [x] **14. DPO Preference Alignment**: Mathematical formulation of DPO.

### Module 7: Applied LLM Engineering & PEFT
- [x] **15. LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning mathematics.

### Module 8: Advanced Architecture
- [x] **16. Mixture of Experts (MoE)**: Routing mathematics for sparse scaling.

## 🛠️ Setup
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---
*Created with ❤️ for the LLM community.*
