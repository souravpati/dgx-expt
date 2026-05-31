"""Overlay attention sweep points on the matmul roofline.

Reads:
  ../roofline/results/roofline.json   (BW + compute roofs)
  ./results/attention.json            (prefill + decode points)

Writes:
  ./results/attention.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOFLINE = HERE.parent / "roofline" / "results" / "roofline.json"
ATTENTION = HERE / "results" / "attention.json"
OUT = HERE / "results" / "attention.png"

SWEEP_STYLE = {
    "prefill_T": dict(color="C0", marker="o", label="prefill (T_q=T_kv=T, causal)"),
    "decode_S":  dict(color="C3", marker="s", label="decode, sweep S (B=1, T_q=1)"),
    "decode_B":  dict(color="C1", marker="^", label="decode, sweep B (S=4096, T_q=1)"),
}

ROOFS_TO_DRAW = ["bf16", "fp8_e4m3"]  # keep the plot uncluttered


def main():
    rl = json.loads(ROOFLINE.read_text())
    att = json.loads(ATTENTION.read_text())

    peak_bw = rl["peak_bandwidth"]["bytes_per_s"]
    peaks = rl.get("peak_compute", {})

    fig, ax = plt.subplots(figsize=(10, 7))
    I_axis = np.logspace(-1, 5, 600)

    # bandwidth diagonal
    ax.loglog(
        I_axis, peak_bw * I_axis / 1e12,
        "k-", lw=1.5, alpha=0.5,
        label=f"BW roof = {peak_bw/1e9:.0f} GB/s",
    )

    # compute roofs (just a couple, the rest add clutter for attention)
    for prec in ROOFS_TO_DRAW:
        peak = peaks.get(prec)
        if not peak or "flops_per_s" not in peak:
            continue
        flops = peak["flops_per_s"]
        knee = flops / peak_bw
        ax.axhline(
            flops / 1e12,
            color="grey", ls="--", lw=1, alpha=0.6,
            label=f"{prec} peak = {flops/1e12:.0f} TF/s (knee {knee:.0f} F/B)",
        )

    # attention sweep points
    for key, style in SWEEP_STYLE.items():
        points = att.get(key, [])
        if not points:
            continue
        xs = [p["intensity_flops_per_byte"] for p in points]
        ys = [p["flops_per_s"] / 1e12 for p in points]
        ax.plot(
            xs, ys,
            ls="-", lw=1.0, ms=6, alpha=0.9,
            **style,
        )

        # annotate the endpoints so the reader can read off T, S, B
        if key == "prefill_T":
            for p in points:
                if p["T_q"] in (64, 4096, 16384):
                    ax.annotate(
                        f"T={p['T_q']}",
                        (p["intensity_flops_per_byte"], p["flops_per_s"] / 1e12),
                        textcoords="offset points", xytext=(6, -2), fontsize=7,
                    )
        if key == "decode_S":
            for p in points:
                if p["T_kv"] in (128, 4096, 32768):
                    ax.annotate(
                        f"S={p['T_kv']}",
                        (p["intensity_flops_per_byte"], p["flops_per_s"] / 1e12),
                        textcoords="offset points", xytext=(6, -2), fontsize=7,
                    )
        if key == "decode_B":
            for p in points:
                if p["B"] in (1, 16, 256):
                    ax.annotate(
                        f"B={p['B']}",
                        (p["intensity_flops_per_byte"], p["flops_per_s"] / 1e12),
                        textcoords="offset points", xytext=(6, 4), fontsize=7,
                    )

    # predicted-decode marker: I = N/K, on the BW diagonal
    shape = att["shape"]
    nk = shape["N_HEADS"] / shape["N_KV_HEADS"]
    bw_at_nk = peak_bw * nk / 1e12
    ax.plot([nk], [bw_at_nk], marker="*", ms=14, color="red", zorder=10,
            label=f"predicted decode (N/K={nk:.0f} F/B, BW-bound)")

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title(
        f"Attention roofline on {att['device']} "
        f"(Llama-3.1-8B shape, bf16, SDPA)"
    )
    ax.set_xlim(0.5, 3e4)
    ax.set_ylim(0.03, 500)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
