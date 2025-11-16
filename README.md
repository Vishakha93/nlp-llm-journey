# nlp-llm-journey

A hands-on repository documenting my journey from **classical NLP concepts**  
(morphemes, tokens, BPE, language models) to **modern LLM systems**  
(transformers, GPU memory, model parallelism, LoRA, and deployment).

This repo is meant to be:
- A learning log
- A code playground (mostly PyTorch + notebooks, often on Google Colab)
- A reference for future system-design / LLM infra interviews

---

## 📚 Structure

### 1. `nlp_foundations/`
Core NLP concepts.

- `notes/` — written notes (Markdown)
  - Morphemes, tokens, subwords
  - BPE / WordPiece
  - Classical language models, n-grams
- `notebooks/` — Colab/Jupyter notebooks for small experiments  
  (tokenization demos, vocabulary growth, etc.)

### 2. `transformer_internals/`
Understanding how transformers work under the hood.

- `notes/` — transformer architecture, attention, positional encodings, etc.
- `notebooks/` — implement attention, transformer blocks, and a tiny GPT-style model.

### 3. `llm_systems/`
How large language models are actually run in production.

- `notes/` — model parameters vs memory, GPU RAM, tensor shapes, parallelism, KV cache, LoRA, deployment.
- `notebooks/` — tensor shape experiments, simple “model parallel” simulations, memory estimation, etc.

### 4. `src/`
Minimal PyTorch code to understand internals (can be imported into notebooks).

- `simple_nn.py` — basic feedforward network with shape printing
- `attention.py` — tiny attention example
- `simple_transformer.py` — skeleton of a transformer block
- `gpt_mini.py` — very small GPT-style model skeleton
- `model_parallel_example.py` — simulate splitting model across “devices”
- `utils.py` — shared helpers (seed setting, device helpers, etc.)

---

## 🚀 Getting Started

> Recommended: use **Google Colab** and clone this repo inside a notebook,  
> or work locally in a virtual environment.

```bash
git clone https://github.com/<your-username>/nlp-llm-journey.git
cd nlp-llm-journey
pip install -r requirements.txt
