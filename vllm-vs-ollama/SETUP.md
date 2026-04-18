# Setup

## Ollama

Installed system-wide (via systemd service).

```bash
ollama pull llama3.1:8b
systemctl status ollama   # verify server is running on :11434
```

Ollama idles with no GPU memory used; it loads the model on first request and unloads after ~5 minutes of inactivity (default).

## vLLM (via Docker)

Docker and the NVIDIA Container Toolkit were preinstalled on the Spark.

### HuggingFace login (one-time)
Llama 3.1 is a gated model. Accept the license on the HuggingFace model page, then:

```bash
pip install huggingface_hub
huggingface-cli login   # paste a Read token
```

Token gets saved to `~/.cache/huggingface/token` and is mounted into the container later.

### Launching vLLM

See [`scripts/start-vllm-docker.sh`](scripts/start-vllm-docker.sh).

```bash
./scripts/start-vllm-docker.sh
```

Image: `vllm/vllm-openai:v0.19.1-cu130` (ARM64 + CUDA 13 build from Docker Hub).

First launch pulls ~8 GB image + downloads model weights (~16 GB) to `~/.cache/huggingface`. Subsequent launches are instant.

## Why Docker instead of pip?

Attempted `pip install vllm` and hit a chain of failures rooted in Blackwell GPU support:

| Step | Error | Cause |
|---|---|---|
| `pip install vllm` | `libcudart.so.12: not found` | Default install pulled CPU-only PyTorch |
| Install torch cu126 | `libtorch_cuda.so: not found` | Reinstalling vllm reverted torch to CPU build |
| Both with `--no-deps` | `cudaErrorNoKernelImageForDevice` | cu126 wheels only ship sm_80/sm_90 kernels; Blackwell is sm_121 |
| Nightly torch 2.12 + vLLM nightly | `undefined symbol: _ZN2at4cuda24getCurrentCUDABlasHandleEv` | ABI mismatch between the two independent nightlies |

The NVIDIA-built container has CUDA 12.8+, Blackwell kernels, and matching torch/vLLM versions all pre-validated. It skips the entire dependency-hell class of problems.

## GPU memory notes

| Framework | GPU memory loaded |
|---|---|
| Ollama (Q4 8B) | ~21 GB |
| vLLM (FP16 8B) | ~110 GB (reserved for KV cache by default) |

vLLM reserves a large fraction of GPU memory upfront for its KV cache, so the two can't run simultaneously on 128 GB unified memory. Stop one before starting the other:

```bash
ollama stop llama3.1:8b      # unload Ollama model
# or Ctrl+C in the vLLM container terminal
```

Verify with:
```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
```
