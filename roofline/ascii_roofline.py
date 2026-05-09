"""ASCII log-log roofline. Pure stdlib, runs on the host."""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).parent / "results" / "roofline.json"

# One char per precision. ASCII view restricts to D=4096 to keep the
# plot readable -- the PNG shows both shapes.
PREC_CHARS = {
    "bf16": "B", "fp16": "F", "fp8_e4m3": "8", "int8": "I",
    "w4a16": "4", "w4a16_marlin": "M",
}
ASCII_SHAPE = "D4096_F4096"


def render(data: dict, width: int = 78, height: int = 24) -> str:
    peak_bw = data["peak_bandwidth"]["bytes_per_s"]
    peaks = data.get("peak_compute", {})

    # Y range: from 10^-1 TF/s to a bit above the highest peak.
    max_tf = max(
        (p["flops_per_s"] / 1e12 for p in peaks.values()
         if isinstance(p, dict) and "flops_per_s" in p),
        default=100,
    )
    x_lo, x_hi = -0.5, 4.0
    y_lo, y_hi = -1.0, math.log10(max_tf) + 0.4

    def to_col(I):
        return int(round((math.log10(I) - x_lo) / (x_hi - x_lo) * (width - 1)))

    def to_row(tf):
        f = (math.log10(tf) - y_lo) / (y_hi - y_lo)
        return (height - 1) - int(round(f * (height - 1)))

    grid = [[" "] * width for _ in range(height)]

    # BW diagonal
    for c in range(width):
        I = 10 ** (x_lo + (x_hi - x_lo) * c / (width - 1))
        tf = peak_bw * I / 1e12
        if tf <= 0 or tf > max_tf * 2:
            continue
        r = to_row(tf)
        if 0 <= r < height and grid[r][c] == " ":
            grid[r][c] = "."

    # Horizontal roof + knee marker per precision
    for prec, peak in peaks.items():
        if not isinstance(peak, dict) or "flops_per_s" not in peak:
            continue
        tf = peak["flops_per_s"] / 1e12
        knee_I = peak["flops_per_s"] / peak_bw
        kc = to_col(knee_I)
        rr = to_row(tf)
        if 0 <= rr < height:
            for c in range(max(0, kc), width):
                if grid[rr][c] == " ":
                    grid[rr][c] = "-"
        if 0 <= kc < width and 0 <= rr < height:
            grid[rr][kc] = "+"

    # Sweep points -- only the D=4096 shape, to keep the ASCII grid readable.
    for label, points in data.get("sweeps", {}).items():
        if not points or not label.startswith(ASCII_SHAPE):
            continue
        prec = label[len(ASCII_SHAPE) + 1:]
        ch = PREC_CHARS.get(prec, "?")
        for p in points:
            r = to_row(p["flops_per_s"] / 1e12)
            c = to_col(p["intensity_flops_per_byte"])
            if 0 <= r < height and 0 <= c < width:
                grid[r][c] = ch

    lines = []
    lines.append(
        f"Roofline -- {data['device']}   "
        f"BW = {peak_bw/1e9:.0f} GB/s"
    )
    peak_summary = ", ".join(
        f"{prec}={p['flops_per_s']/1e12:.0f}TF"
        for prec, p in peaks.items()
        if isinstance(p, dict) and "flops_per_s" in p
    )
    lines.append("peaks: " + peak_summary)
    lines.append("")
    for r in range(height):
        y_log = y_hi - (y_hi - y_lo) * r / (height - 1)
        nearest = round(y_log)
        is_tick = abs(y_log - nearest) < (y_hi - y_lo) / (2 * (height - 1))
        label = f"10^{nearest:+d} " if is_tick else "      "
        lines.append(label + "|" + "".join(grid[r]))
    lines.append("      +" + "-" * width)
    xticks = [" "] * width
    for d in range(int(x_lo), int(x_hi) + 1):
        c = to_col(10 ** d)
        s = f"10^{d:+d}"
        for i, ch in enumerate(s):
            if 0 <= c + i < width:
                xticks[c + i] = ch
    lines.append("       " + "".join(xticks) + "   intensity (F/B)")
    lines.append("")
    lines.append("D=F=4096:  B bf16  F fp16  8 fp8  I int8  4 w4a16  M w4a16_marlin")
    lines.append(". = BW diagonal   - = compute roof   + = knee per precision")
    return "\n".join(lines)


def main():
    data = json.loads(RESULTS.read_text())
    print(render(data))


if __name__ == "__main__":
    main()
