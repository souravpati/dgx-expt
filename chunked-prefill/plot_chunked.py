"""Plot ITL distributions and timelines for chunked vs unchunked vLLM."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OFF = HERE / "results" / "chunked_off.json"
ON = HERE / "results" / "chunked_on.json"
OUT = HERE / "results" / "chunked.png"

PREFILL_SKIP = 20


def itls_and_burst_aligned(d: dict) -> tuple[list[float], list[tuple[float, float]]]:
    """Returns (all ITL samples in ms, list of (t_rel, itl_ms) aligned in time)."""
    bursts = d["burst_events"]
    all_itls_ms: list[float] = []
    timeline: list[tuple[float, float]] = []
    for s in d["streams"]:
        ts = s.get("timestamps_rel", [])
        if len(ts) <= PREFILL_SKIP + 2:
            continue
        ts_eff = ts[PREFILL_SKIP:]
        for j in range(len(ts_eff) - 1):
            dt = ts_eff[j+1] - ts_eff[j]
            all_itls_ms.append(1000 * dt)
            timeline.append((ts_eff[j+1], 1000 * dt))
    return all_itls_ms, timeline


def main():
    if not OFF.exists() or not ON.exists():
        print("Need both results/chunked_off.json and results/chunked_on.json")
        return
    off = json.loads(OFF.read_text())
    on = json.loads(ON.read_text())

    off_itls, off_timeline = itls_and_burst_aligned(off)
    on_itls, on_timeline = itls_and_burst_aligned(on)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- top-left: ITL distribution (log-scaled, histogram)
    ax = axes[0, 0]
    bins = np.logspace(np.log10(10), np.log10(max(off_itls + on_itls, default=10000)), 80)
    ax.hist(off_itls, bins=bins, alpha=0.55, color="C3", label="chunked OFF")
    ax.hist(on_itls,  bins=bins, alpha=0.55, color="C0", label="chunked ON")
    ax.set_xscale("log")
    ax.set_xlabel("ITL (ms)")
    ax.set_ylabel("count")
    ax.set_title("Inter-token latency distribution (log x)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # --- top-right: tail-percentile bars
    ax = axes[0, 1]
    pcts = ["p50", "p90", "p95", "p99", "p99_9"]
    off_vals = [off["stats"].get(f"ms_{p}", 0) for p in pcts]
    on_vals = [on["stats"].get(f"ms_{p}", 0) for p in pcts]
    x = np.arange(len(pcts))
    w = 0.4
    ax.bar(x - w/2, off_vals, w, color="C3", label="chunked OFF")
    ax.bar(x + w/2, on_vals,  w, color="C0", label="chunked ON")
    for i, (a, b) in enumerate(zip(off_vals, on_vals)):
        ax.text(i - w/2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", ".") for p in pcts])
    ax.set_yscale("log")
    ax.set_ylabel("ITL (ms, log)")
    ax.set_title("ITL percentiles")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # --- bottom-left: ITL over time, OFF
    ax = axes[1, 0]
    if off_timeline:
        ts, vs = zip(*off_timeline)
        ax.scatter(ts, vs, s=2, alpha=0.4, color="C3")
    for b in off["burst_events"]:
        ax.axvline(b["t_rel"], color="grey", ls=":", lw=0.7, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ITL (ms, log)")
    ax.set_title(f"OFF: ITL timeline (dotted lines = prefill bursts, n={len(off['burst_events'])})")
    ax.grid(True, which="both", alpha=0.3)

    # --- bottom-right: ITL over time, ON
    ax = axes[1, 1]
    if on_timeline:
        ts, vs = zip(*on_timeline)
        ax.scatter(ts, vs, s=2, alpha=0.4, color="C0")
    for b in on["burst_events"]:
        ax.axvline(b["t_rel"], color="grey", ls=":", lw=0.7, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ITL (ms, log)")
    ax.set_title(f"ON: ITL timeline (n bursts={len(on['burst_events'])})")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Chunked prefill: does interleaving prefill protect decode latency?",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
