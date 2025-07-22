# PyTorch Expert Demo Repo

This repository is a comprehensive demonstration of advanced PyTorch usage — spanning from low-level autograd mechanics to multi-GPU distributed training, PEFT-based LLM fine-tuning, and custom CUDA ops. Each module is designed to highlight deep system-level knowledge, research-level implementation skills, and production-grade deployment experience.

## 🔥 Modules Overview

### 1. `autograd_engine/`
Build a minimal PyTorch-style `Tensor` class with automatic differentiation. Implements basic operators and backward propagation logic from scratch.

> ✅ Highlights: Graph construction, gradient checking, fundamental understanding of `autograd` internals.

---

### 2. `distributed_training/`
Train a ResNet/ViT model using:
- `torch.distributed` with `torchrun`
- `DistributedDataParallel`
- `FullyShardedDataParallel (FSDP)`

Includes profiler logs, memory usage comparison, and training speedup graphs.

> ✅ Highlights: Multi-GPU scaling, FSDP internals, fault tolerance, profiling.

---

### 3. `custom_llm_peft/`
Train a mini Transformer-based language model (~100M params) and fine-tune it using:
- LoRA / QLoRA
- PEFT integration
- Memory benchmarking vs vanilla fine-tuning

> ✅ Highlights: PEFT techniques, model memory efficiency, HF `transformers` + `peft` ecosystem.

---

### 4. `rl_torchscript/`
Reinforcement learning with a custom `gym.Env`:
- Proximal Policy Optimization (PPO)
- TorchScript export and simulated production inference
- Real-time visualization

> ✅ Highlights: RL pipeline, model scripting, real-time interaction.

---

### 5. `custom_cuda_op/`
Write a custom CUDA kernel for a normalization operator (e.g., GroupNorm) and integrate it with PyTorch using `cpp_extension`.

> ✅ Highlights: PyTorch C++/CUDA extension interface, performance comparison with native ops.

---

## 📦 Planned Add-ons (Coming Soon)
- `performance_tuning/`: Compare TorchScript, TorchDynamo, and FP16/BF16 auto casting.
- `nas_from_scratch/`: Differentiable NAS framework for CNNs.
- `paper_reproduction/`: Reproduction of DINOv2 or RWKV.
- `fastapi_deployment/`: FastAPI inference endpoint with TorchServe integration.
- `quantization_edge/`: ONNX export, quantization, and deployment to Jetson Nano.

---

## 🧠 Target Audience
- LLM Engineers
- ML Infra/Platform Engineers
- Research Engineers
- AI DevOps Specialists

---

## 🚀 Quickstart
```bash
# Clone and install requirements
$ git clone https://github.com/codefortune-cook/torchcraft
$ cd torchcraft
$ pip install -r requirements.txt

# Run a module, e.g.,
$ cd 1_autograd_engine
$ python train.py
```

---

## 🛠️ Environment
Tested on:
- PyTorch 2.3+
- CUDA 12.1 / ROCm 6.0 (where applicable)
- Python 3.10+
- Multi-GPU setups (A100 x 4, 3090 x 2, or T4 x 4)

---

## License
MIT
