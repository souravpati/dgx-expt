"""Plot TTFT vs turn for the growing-context multi-turn sweep."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ON = HERE / "results" / "multiturn_on.json"
OFF_REF = HERE / "results" / "prefix_off.json"
OUT = HERE / "results" / "multiturn.png"


def main():
    d = json.loads(ON.read_text())
    rows = d["rows"]
    turns = [r["turn"] for r in rows]
    prompt_lens = [r["prompt_len"] for r in rows]
    ttfts = [r["ttft_s"] * 1000 if r.get("ttft_s") else float("nan") for r in rows]

    # Extrapolate "no cache" TTFT from prefix_off (957 ms for 4160 tokens)
    if OFF_REF.exists():
        off = json.loads(OFF_REF.read_text())
        warm_means = [r["ttft_s"] * 1000 for r in off["rows"][1:] if r.get("ttft_s")]
        off_ttft_at_4160 = float(np.mean(warm_means))
        per_1024 = off_ttft_at_4160 / 4160 * 1024
        off_pred = [per_1024 * L / 1024 for L in prompt_lens]
    else:
        off_pred = None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(turns, ttfts, marker="o", ms=10, lw=2, color="C0",
            label="caching ON (measured)")
    if off_pred:
        ax.plot(turns, off_pred, marker="x", ms=10, lw=1.5, ls="--",
                color="C3", label="caching OFF (extrapolated)")

    for t, lt, ttft in zip(turns, prompt_lens, ttfts):
        ax.annotate(f"{ttft:.0f} ms\n({lt} tok)", (t, ttft),
                    textcoords="offset points", xytext=(6, 8), fontsize=8)

    ax.set_xlabel("Turn (each adds 1024 tokens of new context)")
    ax.set_ylabel("Time to first token (ms)")
    ax.set_title("Multi-turn prefix caching: TTFT stays flat as context grows\n"
                 "Llama 3.1 8B Instruct on GB10")
    ax.set_xticks(turns)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
