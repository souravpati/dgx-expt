"""Overlay measured ms/step on the predicted curve.

Reads:
  results/predicted_step_time.json
  results/measured_step_time.json   (if present)
Writes:
  results/step_time.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
PRED = HERE / "results" / "predicted_step_time.json"
MEAS = HERE / "results" / "measured_step_time.json"
OUT = HERE / "results" / "step_time.png"

S_COLORS = {2048: "C0", 8192: "C1", 32768: "C3"}


def main():
    pred = json.loads(PRED.read_text())
    bcrit = pred["constants"]["B_crit"]

    pred_by_S = defaultdict(list)
    for r in pred["rows"]:
        pred_by_S[r["S"]].append(r)
    for S in pred_by_S:
        pred_by_S[S].sort(key=lambda r: r["B"])

    meas_by_S = defaultdict(list)
    if MEAS.exists():
        meas = json.loads(MEAS.read_text())
        for r in meas["rows"]:
            if "ms_per_step" in r:
                meas_by_S[r["S"]].append(r)
        for S in meas_by_S:
            meas_by_S[S].sort(key=lambda r: r["B"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for S, color in S_COLORS.items():
        rows = pred_by_S.get(S, [])
        if not rows:
            continue
        xs = [r["B"] for r in rows]
        ys_step = [r["ms_per_step"] for r in rows]
        ys_per_tok = [r["ms_per_token"] for r in rows]
        ax1.plot(xs, ys_step, "-", color=color, lw=1.2,
                 label=f"predicted S={S}")
        ax2.plot(xs, ys_per_tok, "-", color=color, lw=1.2,
                 label=f"predicted S={S}")

        meas = meas_by_S.get(S, [])
        if meas:
            mxs = [r["B"] for r in meas]
            mys_step = [r["ms_per_step"] for r in meas]
            mys_per_tok = [r["ms_per_token_aggregate"] for r in meas]
            ax1.plot(mxs, mys_step, "o", color=color, ms=8,
                     label=f"measured S={S}")
            ax2.plot(mxs, mys_per_tok, "o", color=color, ms=8,
                     label=f"measured S={S}")

    for ax in (ax1, ax2):
        ax.axvline(bcrit, color="grey", ls="--", lw=1, alpha=0.7,
                   label=f"B_crit = {bcrit:.0f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch B (concurrent requests)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    ax1.set_ylabel("ms / step (= user-visible ITL)")
    ax1.set_title("Step latency vs batch size")
    ax2.set_ylabel("ms / token (aggregate, = step / B)")
    ax2.set_title("Per-token latency from cluster POV")

    suptitle = (
        f"Llama 3.1 8B bf16 on GB10  |  "
        f"P=16GB, W_HBM=245 GB/s, W_FLOPs=90 TF/s, KV/tok=4 KiB"
    )
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
