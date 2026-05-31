# Cleanup Guide

Disk usage accumulates in three places when running these experiments. None of it is urgent, but here's how to check and clean each.

## Where disk is used

| Location | Typical size | Persisted across reboots? | Purpose |
|----------|--------------|---------------------------|---------|
| `/tmp/*_trace.csv`, `/tmp/*.log` | KB–MB | **No** (systemd-tmpfiles cleans after 10 days by default) | Benchmark sample traces |
| `/var/lib/docker/` (images) | ~20 GB | Yes | vLLM container image |
| `/var/lib/docker/overlay2/` (containers) | tens of MB | Reclaimed when container stops with `--rm` | Per-container writeable layer |
| `~/.cache/huggingface/hub/` | ~16 GB | Yes | Model weights |

## Check current usage

```bash
# Trace files
ls -la /tmp/*trace* /tmp/*usage* 2>/dev/null

# HuggingFace model cache
du -sh ~/.cache/huggingface/

# Docker storage (images, containers, volumes, build cache)
sudo docker system df
sudo du -sh /var/lib/docker

# Overall disk
df -h /
```

## Clean individual items

### Trace files — safe to remove anytime
```bash
rm -f /tmp/vllm_kv_trace.csv /tmp/ollama_kv_trace.csv \
      /tmp/gpu_usage_vllm.log /tmp/gpu_usage_ollama.log \
      /tmp/vllm_kv_scenario /tmp/ollama_kv_scenario \
      /tmp/vllm_resp_*.json /tmp/ollama_resp_*.json
```

### Docker — stopped containers, dangling images, unused volumes
```bash
# Safe — reclaims anything not in active use
sudo docker system prune

# Aggressive — also removes images with no running container
# (will force re-pull of vllm image next run, ~20 GB download)
sudo docker system prune -a
```

### HuggingFace model cache — removes Llama 3.1 weights
```bash
# Forces a ~16 GB re-download next time you start vLLM
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
```

### Ollama models
```bash
ollama list                  # see what's pulled
ollama rm llama3.1:8b        # ~5 GB freed
```

### GPU memory (in-flight processes, not disk)
```bash
# Unload Ollama model without stopping the service
ollama stop llama3.1:8b

# Stop vLLM container (also reclaims its writeable layer)
# Press Ctrl+C in the terminal where it's running, or:
sudo docker ps          # find container ID
sudo docker stop <id>
```

## About not rebooting often

DGX Spark is designed to stay on. Running for weeks or months is fine — a few considerations:

### What accumulates without reboot
- **`/tmp` files** — systemd-tmpfiles cleans files untouched for >10 days automatically, so old traces evaporate on their own
- **journal logs** (`/var/log/journal/`) — systemd rotates these (default limit ~1 GB)
- **Docker writeable layers** — `--rm` on `docker run` handles this; no buildup from our scripts
- **HuggingFace/pip caches** — only grow when you download something new
- **Kernel memory leaks** — rare on modern kernels, but possible over months

### What matters more than uptime

| Concern | Mitigation |
|---------|------------|
| Security/kernel patches | `sudo apt update && sudo apt upgrade` weekly, reboot when a kernel update comes in |
| NVIDIA driver updates | Only require reboot when kernel driver version changes |
| GPU in weird state after crashes | `sudo nvidia-smi --gpu-reset` can fix it without rebooting |
| Disk filling up | Use the commands above to check; unlikely to happen just from experiments |

### Practical monitoring

```bash
# Quick health snapshot
uptime                          # how long it's been up
df -h /                         # root disk usage
free -h                         # memory pressure (watch available column)
sudo docker system df           # docker disk claim
```

If `free -h` shows very low available memory after lots of vLLM runs (you stopped the container but memory didn't come back), that's the main reboot-worthy situation — unlikely but possible due to driver/container interactions. Usually `sudo systemctl restart docker` fixes it without a full reboot.

### TL;DR

You don't need to reboot regularly. Just:
1. Run `sudo apt upgrade` monthly and reboot when a new kernel is staged
2. Watch `df -h /` occasionally so you don't fill disk
3. Run `sudo docker system prune` every few weeks if you've pulled lots of images
