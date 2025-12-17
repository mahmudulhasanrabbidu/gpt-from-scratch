# GPT From Scratch

This repository contains a **from-scratch implementation of a GPT-style language model**, built end-to-end to understand how modern language models work at a fundamental level.

Everything is implemented manually — from tokenization to the Transformer architecture, training loop, and autoregressive text generation — with minimal reliance on high-level abstractions.

The goal of this project is **learning and understanding**, not performance or scale.


## Motivation

Large language models are often treated as black boxes.  
This project exists to answer deeper questions:

- How does a tokenizer shape learning?
- How does attention actually work, step by step?
- What does it mean for a model to *learn* patterns in text?
- How far can intelligence be built from first principles?
- This project serves as a foundational step in my broader goal: **Exploring the foundations of the human brain and emotion to solve AGI.** By rebuilding these systems from the ground up, I aim to demystify the "black box" of modern AI.

This repository is part of a broader personal effort to explore **AGI, human cognition, emotion, and biological learning** through implementation and experimentation.



## What Is Implemented From Scratch

- **Tokenizer**
  - Custom tokenizer (regex / BPE-style)
  - Vocabulary construction
  - Encoding and decoding logic

- **Model Architecture**
  - Token and positional embeddings
  - Multi-head self-attention
  - Causal masking
  - Feed-forward networks
  - Residual connections and layer normalization
  - Full Transformer decoder stack

- **Training**
  - Custom training loop
  - Loss computation
  - Optimizer configuration
  - Device handling (CPU / CUDA)
  - Progress logging

- **Text Generation**
  - Autoregressive sampling
  - Temperature and top-k sampling
  - Context window handling


