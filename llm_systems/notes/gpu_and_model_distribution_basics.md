# GPU Memory, Model Parameters & How Large Models Run Across Multiple GPUs
*A foundational, intuitive overview based on interactive conversation with chatGPT.*

---

## 1. What are model parameters?

Model parameters are **just numbers** (floats) that the model has learned during training.

- Each weight matrix contains many such numbers.
- A GPT-style model with **175B parameters** literally stores 175 billion numbers.
- These numbers must live somewhere **fast enough** for matrix multiplications.

### Parameter sizes
| Precision | Bytes per parameter |
|----------|----------------------|
| FP32     | 4 bytes              |
| BF16/FP16| 2 bytes              |
| INT8     | 1 byte               |

### Example

**175B parameters × 2 bytes (FP16) ≈ 350 GB**  
This size *cannot* fit into a single GPU’s 40GB or 80GB VRAM.

---

## 2. Where do parameters live?

All model parameters must be stored in **GPU VRAM** during inference or training.

Why not CPU RAM?

- GPU compute cores need **local**, **fast**, **contiguous** memory access.
- Fetching parameters from CPU RAM over PCIe would be **hundreds of times slower**.

Thus, parameters must reside inside GPU memory.

---

## 3. Why do we need multiple GPUs?

A single GPU does **not** have enough memory to store:

- Model weights (e.g., 350GB)
- Intermediate activations
- Attention KV cache
- Temporary buffers for matrix multiplication

### Example

| Model | FP16 Size | GPUs Needed |
|-------|-----------|--------------|
| GPT-3 175B | ~350 GB | 8–16 GPUs |
| Llama-2 70B | 140 GB | 4–8 GPUs |

We must **split the model across GPUs** to make it runnable.

---

## 4. How are LLMs split across GPUs?

There are two main strategies.

---

### A. Tensor Parallelism (splitting *within* a layer)

A single layer’s weight matrix is divided across multiple GPUs.

Example:  
A matrix of shape `[4096 × 4096]` is split into:

GPU0 → columns 0–2047
GPU1 → columns 2048–4095

Each GPU computes part of the result, then results are combined.

Used when **one layer is too big** for one GPU.

---

### B. Pipeline Parallelism (splitting *layers* across GPUs)

Imagine a 48-layer transformer:

- GPU0 → Layers 1–12  
- GPU1 → Layers 13–24  
- GPU2 → Layers 25–36  
- GPU3 → Layers 37–48  

This forms an **assembly line**:

Input → GPU0 → GPU1 → GPU2 → GPU3 → Output

Each GPU only stores its portion of the model.

---

### In practice: both are used

Large-scale systems (DeepSpeed, Megatron-LM, TPU Pods) combine:

- Tensor parallelism  
- Pipeline parallelism  
- Data parallelism (for throughput)

---

## 5. What is the “tensor” passed between layers?

A **tensor** is a multi-dimensional array produced by each layer.

Typical shapes:

- `(batch, seq_len, hidden_dim)`
- e.g., `(1, 200, 4096)`

Each layer transforms the tensor and passes it forward.

### Shape validation  
If a layer receives a mismatched shape, PyTorch/TF raises an error.

Transformers are architected so all shapes are known **in advance**.

---

## 6. How do GPUs share tensors?

GPUs *do not* communicate using APIs or network calls.  
Instead, they use hardware-level, extremely fast links:

- **NVLink**
- **NVSwitch**
- **PCIe**
- **NCCL** (NVIDIA Collective Communications Library)

These support:

- `all_reduce`
- `broadcast`
- `scatter` / `gather`
- `reduce_scatter`

Data moves between GPUs at hundreds of GB/s.

---

## 7. How ChatGPT uses many GPUs when you send a message

### 1. Tokenization (CPU)
Your text → token IDs (e.g., `[1542, 89, 472, ...]`)

### 2. Embedding (GPU)
Token IDs → embedding vectors.

### 3. Transformer blocks run across GPUs
- Some layers live on GPU0
- Some on GPU1
- Some on GPU2
- …
- Or single layers are split across GPUs (tensor parallel)

### 4. GPUs exchange intermediate tensors
Using NVLink/NCCL collectives.

### 5. Final logits → token
LLM outputs probability distribution → next token.

---

## 8. Why deploying one 175B model is expensive

Each replica requires:

- 8–16 high-end GPUs (A100/H100)
- 350GB of VRAM
- High-speed interconnect fabric
- Autoscaling infrastructure
- Replicas for load balancing

### Cost example
1 A100 → $2–$3/hour  
8 A100s → $16–$24/hour  
≈ ~$15,000/month **per model instance**

Multiple fine-tuned variants multiply cost linearly.

---

## 9. Multi-GPU Inference Server (High-Level Architecture)

Below is a diagram showing how an inference request flows across CPU → multi-GPU pipeline → API/UI.

<p align="center">
  <img src="../images/multi_gpu_inference.png" alt="Multi-GPU Inference Server Architecture" width="700"/>
</p>

**Explanation:**

- **CPU**
  - Handles tokenization and request routing.
  - Prepares input tensors before sending to GPU pipeline.

- **GPU Pipeline (Pipeline Parallelism)**
  - Model layers split across GPUs (GPU0 → GPU1 → GPU2 → GPU3).
  - Each GPU processes its segment of layers and forwards activations.

- **UI/API**
  - Exposes REST endpoints.
  - Streams generated tokens back to the user.

---


