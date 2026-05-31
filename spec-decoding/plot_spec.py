"""Plot tokens/sec vs K (num_speculative_tokens), prose vs code."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results" / "spec_decoding.png"

KS = [0, 3, 5]


def load(k):
    p = HERE / "results" / f"spec_K{k}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    datas = {k: load(k) for k in KS}
    available = [k for k, d in datas.items() if d is not None]
    if not available:
        print("No spec_K*.json files found yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # tokens/sec uses wall + n_streamed_chunks; we also have spec_emitted
    # for K>=1. For K=0 we approximate true tokens = chunks.
    # Better: pull generation_tokens_total_delta to get a clean count.

    for ax_idx, work in enumerate(["prose", "code"]):
        ax = axes[ax_idx]
        ks_x, tps_y, baselines, accepts = [], [], [], []
        for k in KS:
            d = datas.get(k)
            if not d: continue
            w = d.get(work, {})
            n_gen = w.get("generation_tokens_total_delta")
            mean_wall = w.get("mean_wall_s")
            mean_chunks = w.get("mean_chunks_per_s")
            acc = w.get("spec_acceptance_rate")
            # use mean_wall + assumed token count per prompt = max_output (256)
            n_prompts = w.get("n_prompts", 0)
            if n_gen and mean_wall:
                tps = n_gen / (mean_wall * n_prompts)
            elif mean_chunks:
                tps = mean_chunks  # chunks/s with spec >= 1 chunk per token
            else:
                continue
            ks_x.append(k)
            tps_y.append(tps)
            accepts.append(acc)
        if not ks_x: continue

        ax.plot(ks_x, tps_y, marker="o", ms=10, lw=2,
                color="C0" if work == "prose" else "C3",
                label=f"{work} measured")

        baseline = tps_y[0] if ks_x[0] == 0 else None
        if baseline:
            # predicted theoretical max: K * baseline
            ax.plot(ks_x, [baseline * max(1, k) for k in ks_x], "k--",
                    alpha=0.5, label="theoretical max (K * baseline)")

        for k, tps, acc in zip(ks_x, tps_y, accepts):
            label = f"{tps:.1f} tok/s"
            if acc is not None:
                label += f"\nα={acc:.2f}"
            ax.annotate(label, (k, tps), textcoords="offset points",
                        xytext=(8, -8), fontsize=8)

        ax.set_xlabel("K = num_speculative_tokens")
        ax.set_ylabel("Tokens/sec (single user)")
        ax.set_title(f"{work}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(KS)

    fig.suptitle(
        "Speculative decoding: tok/s vs K  |  Llama-3.1-8B target + Llama-3.2-1B draft  |  GB10",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
