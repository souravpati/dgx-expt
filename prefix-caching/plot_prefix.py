"""Plot TTFT per request, OFF vs ON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OFF = HERE / "results" / "prefix_off.json"
ON = HERE / "results" / "prefix_on.json"
OUT = HERE / "results" / "prefix_caching.png"


def ttfts(d: dict) -> list[float]:
    return [r["ttft_s"] * 1000 if r.get("ttft_s") else float("nan")
            for r in d["rows"]]


def main():
    if not OFF.exists() or not ON.exists():
        print("Need both results/prefix_off.json and results/prefix_on.json")
        return
    off = json.loads(OFF.read_text())
    on = json.loads(ON.read_text())

    off_t = ttfts(off)
    on_t = ttfts(on)
    xs = np.arange(len(off_t))

    fig, ax = plt.subplots(figsize=(11, 6))
    w = 0.4
    bars_off = ax.bar(xs - w/2, off_t, w, color="C3", label="prefix cache OFF")
    bars_on = ax.bar(xs + w/2, on_t, w, color="C0", label="prefix cache ON")

    # Annotate values
    for x, v in zip(xs - w/2, off_t):
        if not np.isnan(v):
            ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for x, v in zip(xs + w/2, on_t):
        if not np.isnan(v):
            ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"req {i+1}" for i in xs])
    ax.set_ylabel("Time to first token (ms)")
    cfg = off["config"]
    ax.set_title(
        f"Prefix caching: shared prefix={cfg['shared_prefix_len']} tokens, "
        f"query={cfg['query_len']}, output={cfg['output_tokens']}\n"
        f"Llama 3.1 8B Instruct on GB10"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    # Summary text
    off_mean_rest = float(np.nanmean(off_t[1:]))
    on_mean_rest = float(np.nanmean(on_t[1:]))
    on_first = on_t[0]
    speedup = off_mean_rest / on_mean_rest if on_mean_rest else float("nan")
    ax.text(
        0.02, 0.98,
        f"OFF: mean TTFT (req 2-10) = {off_mean_rest:.0f} ms\n"
        f"ON:  req 1 (cold) = {on_first:.0f} ms\n"
        f"ON:  mean TTFT (req 2-10) = {on_mean_rest:.0f} ms\n"
        f"Speedup on warm requests: {speedup:.1f}x",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"),
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
