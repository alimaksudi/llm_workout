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
- [ ] **04. Positional Encoding**: Injecting sequence order (Sin/Cos vs. RoPE).
- [ ] **05. Attention Mechanisms**: Scaled Dot-Product, Self-Attention, and Multi-Head Attention.

### Module 3: Architecture Assembly
- [ ] **06. The Encoder Layer**: Residual connections, LayerNorm, and FFN.
- [ ] **07. The Decoder Layer**: Masked Attention and Cross-Encoder Attention.
- [ ] **08. Full Transformer**: Putting it all together.

### Module 4: Efficiency & Scaling
- [ ] **09. KV Caching**: Optimizing inference speed.
- [ ] **10. Advanced Attention**: GQA (Grouped Query) and MQA (Multi-Query).

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
