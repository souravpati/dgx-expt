"""Plot GQA-ratio sweep: decode TF/s vs N/K, with predicted W_HBM*N/K line."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "results" / "gqa_sweep.json"
OUT = HERE / "results" / "gqa_sweep.png"

ARCH_LABELS = {
    1: "MHA\n(Llama 2 7B)",
    2: "GQA-16",
    4: "GQA-8\n(Llama 3.1 8B)",
    8: "GQA-4\n(Llama 3 70B)",
    16: "GQA-2",
    32: "MQA",
}


def main():
    d = json.loads(DATA.read_text())
    peak_bw = d["peak_bw_gb_per_s"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Predicted: TF/s = peak_bw * N/K (decode is BW-bound, intensity = N/K)
    nk_axis = np.linspace(0.8, 40, 200)
    pred_tflops = peak_bw * nk_axis / 1000  # GB/s * F/B -> GFLOPs/s; /1000 = TF/s
    ax1.plot(nk_axis, pred_tflops, "k--", lw=1.2, alpha=0.6,
             label=f"predicted: W_HBM * N/K  ({peak_bw:.0f} GB/s * N/K)")

    # Achieved bw line at peak
    ax2.axhline(peak_bw, color="k", ls="--", lw=1, alpha=0.5,
                label=f"peak BW = {peak_bw:.0f} GB/s")

    by_B = {}
    for r in d["rows"]:
        if "error" in r:
            continue
        by_B.setdefault(r["B"], []).append(r)

    colors = {1: "C0", 8: "C3"}
    for B in sorted(by_B.keys()):
        rows = sorted(by_B[B], key=lambda r: r["n_over_k"])
        xs = [r["n_over_k"] for r in rows]
        ys_tf = [r["tflops_per_s"] for r in rows]
        ys_gb = [r["gb_per_s"] for r in rows]
        ax1.plot(xs, ys_tf, marker="o", ms=8, lw=1.2,
                 color=colors.get(B, "C2"), label=f"measured B={B}")
        ax2.plot(xs, ys_gb, marker="o", ms=8, lw=1.2,
                 color=colors.get(B, "C2"), label=f"measured B={B}")

    # Annotate known architectures along the predicted line
    for nk, label in ARCH_LABELS.items():
        ax1.annotate(label, (nk, peak_bw * nk / 1000),
                     textcoords="offset points", xytext=(6, -28),
                     fontsize=7, color="grey")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("GQA ratio  N/K  (= decode arithmetic intensity, F/B)")
    ax1.set_ylabel("Achieved decode TF/s")
    ax1.set_title("Decode throughput scales linearly with GQA ratio")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("GQA ratio N/K")
    ax2.set_ylabel("Achieved bandwidth (GB/s)")
    ax2.set_title("Achieved DRAM bandwidth (should be flat at peak)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    shape = d["fixed"]
    fig.suptitle(
        f"GQA ratio sweep on {d['device']}  |  "
        f"N={shape['N_HEADS']}, H={shape['HEAD_DIM']}, T_kv={shape['T_q']}->{shape['T_kv']}, bf16",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
