"""Plot the empirical roofline from results/roofline.json.

Renders a PNG with one diagonal (bandwidth) and one horizontal roof
per precision, plus the per-shape sweeps as scatter lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results" / "roofline.json"
OUT = Path(__file__).parent / "results" / "roofline.png"

PRECISION_COLORS = {
    "bf16":         "C0",
    "fp16":         "C2",
    "fp8_e4m3":     "C3",
    "int8":         "C4",
    "w4a16":        "C1",
    "w4a16_marlin": "C5",
}

SHAPE_MARKERS = {
    "D4096_F4096": "o",
    "D1024_F1024": "s",
}


def main():
    data = json.loads(RESULTS.read_text())

    peak_bw = data["peak_bandwidth"]["bytes_per_s"]
    peaks = data.get("peak_compute", {})

    fig, ax = plt.subplots(figsize=(10, 7))
    I_axis = np.logspace(-1, 4, 400)

    # One diagonal (BW-bound) line; same for every precision because the
    # diagonal in F/B space is just bytes/s -- it's the bytes-per-elem
    # accounting that puts each precision at a different intensity for
    # the same shape.
    ax.loglog(I_axis, peak_bw * I_axis / 1e12, "k-", lw=1.5, alpha=0.5,
              label=f"BW roof = {peak_bw/1e9:.0f} GB/s")

    # Horizontal compute roof per precision.
    for prec, peak in peaks.items():
        if not peak or "flops_per_s" not in peak:
            continue
        flops = peak["flops_per_s"]
        knee = flops / peak_bw
        ax.axhline(flops / 1e12, color=PRECISION_COLORS.get(prec, "grey"),
                   ls="--", lw=1, alpha=0.7,
                   label=f"{prec} peak = {flops/1e12:.0f} TF/s (knee {knee:.0f} F/B)")

    # Sweep points.
    for label, points in data.get("sweeps", {}).items():
        if not points:
            continue
        # label is "<shape>_<prec>", e.g. "D4096_F4096_bf16"
        shape = "_".join(label.split("_")[:2])
        prec = label[len(shape) + 1:]
        color = PRECISION_COLORS.get(prec, "grey")
        marker = SHAPE_MARKERS.get(shape, "x")
        xs = [p["intensity_flops_per_byte"] for p in points]
        ys = [p["flops_per_s"] / 1e12 for p in points]
        ax.plot(xs, ys, marker=marker, ls="-", color=color,
                ms=4, lw=0.8, alpha=0.85,
                label=f"{shape} {prec}")

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title(f"Roofline by precision -- {data['device']}")
    ax.set_xlim(0.3, 1e4)
    ax.set_ylim(0.05, 1500)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
