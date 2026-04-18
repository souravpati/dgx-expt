# vLLM vs Ollama — Latency & Throughput on DGX Spark

Compare single-request latency and concurrent throughput of two LLM serving frameworks on the same hardware and model.

## Hardware

- **NVIDIA DGX Spark** (GB10 Grace Blackwell superchip)
- 20-core ARM CPU, Blackwell GPU, ~128 GB unified LPDDR5X
- Compute capability: sm_121
- OS: Ubuntu 24.04 (aarch64)
- CUDA: 13.0 system, 12.8 inside container (for Blackwell kernels)

## Model

`meta-llama/Meta-Llama-3.1-8B-Instruct`
- Ollama: `llama3.1:8b` tag, Q4_K_M quantization (4-bit)
- vLLM: FP16 precision (16-bit)

These are **not apples-to-apples** — vLLM carries 4× more data per token. The test captures the real difference a user would experience with each tool's defaults.

## Docs

- [`SETUP.md`](SETUP.md) — installation and environment
- [`EXPERIMENT.md`](EXPERIMENT.md) — test methodology
- [`RESULTS.md`](RESULTS.md) — numbers and analysis

## Scripts

- [`scripts/start-vllm-docker.sh`](scripts/start-vllm-docker.sh) — launch vLLM container
- [`scripts/bench-single-ollama.sh`](scripts/bench-single-ollama.sh) — Ollama sequential latency
- [`scripts/bench-single-vllm.sh`](scripts/bench-single-vllm.sh) — vLLM sequential latency
- [`scripts/bench-concurrent-ollama.sh`](scripts/bench-concurrent-ollama.sh) — Ollama 10 parallel requests
- [`scripts/bench-concurrent-vllm.sh`](scripts/bench-concurrent-vllm.sh) — vLLM 10 parallel requests

## TL;DR

- **Single user, lowest latency, easy setup** → Ollama with quantization
- **Multi-user API serving, high GPU utilization** → vLLM (scales with load; Ollama doesn't)
- **Quantization is the biggest knob for single-request latency**
- **Blackwell is still early-adopter territory for pip wheels** — containers are the pragmatic path
