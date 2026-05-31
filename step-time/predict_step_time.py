"""
Predicted decode-step latency vs batch, using Chapter-7 formula.

Plugs in numbers measured by sibling experiments:
  - P (params)   = 16 GB for Llama 3.1 8B in bf16
  - KV/token     = 128 KiB (4 KiB/layer x 32 layers; the engine reads
                            all layers' KV cache every decode step)
  - W_HBM        = 213 GB/s   (clean 1-GiB d2d copy from roofline/;
                                attention measured 225-232 GB/s with
                                some L2 leak -- 213 is the conservative
                                hardware ceiling)
  - W_FLOPs      = 90 TF/s    (bf16 peak from roofline)
  - B_crit       = 367        (W_FLOPs / W_HBM)

No GPU needed. Writes results/predicted_step_time.json so the plot
script can overlay measurements on top.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "results" / "predicted_step_time.json"

# --- model + hardware constants ---
P_BYTES = 16e9
L_LAYERS = 32
KV_PER_TOKEN = 2 * 8 * 128 * 2 * L_LAYERS  # 128 KiB (per-layer 4 KiB x 32 layers)
W_HBM = 213e9
W_FLOPS = 90e12
B_CRIT = W_FLOPS / W_HBM        # 367.3
T_MLP_BW_FLOOR = P_BYTES / W_HBM  # param-load time, lower bound for MLP


def step_seconds(B: int, S: int) -> dict:
    kv_total = B * KV_PER_TOKEN * S
    t_attn = kv_total / W_HBM
    t_mlp_flops = 2 * B * P_BYTES / W_FLOPS
    t_mlp = max(T_MLP_BW_FLOOR, t_mlp_flops)
    t_total = t_attn + t_mlp
    return {
        "B": B, "S": S,
        "t_attn_s": t_attn,
        "t_mlp_s": t_mlp,
        "t_mlp_regime": "compute" if t_mlp_flops > T_MLP_BW_FLOOR else "bandwidth",
        "t_total_s": t_total,
        "ms_per_step": t_total * 1000,
        "ms_per_token": t_total * 1000 / B,
    }


def main():
    Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 367, 512, 1024]
    Ss = [2048, 8192, 32768]

    rows = []
    for S in Ss:
        for B in Bs:
            rows.append(step_seconds(B, S))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "constants": {
            "P_bytes": P_BYTES,
            "KV_per_token_bytes": KV_PER_TOKEN,
            "W_HBM_bytes_per_s": W_HBM,
            "W_FLOPs_per_s": W_FLOPS,
            "B_crit": B_CRIT,
            "t_mlp_bw_floor_s": T_MLP_BW_FLOOR,
        },
        "rows": rows,
    }, indent=2))
    print(f"Wrote {OUT}")

    # Pretty table
    print(f"\nB_crit = {B_CRIT:.0f}  (MLP turns compute-bound)")
    print(f"\n{'B':>5}  " + "  ".join(f"S={S:>5} ms/step" for S in Ss)
          + "  || " + "  ".join(f"S={S:>5} ms/tok" for S in Ss))
    for B in Bs:
        line = f"{B:>5}  "
        line += "  ".join(
            f"{next(r for r in rows if r['B']==B and r['S']==S)['ms_per_step']:>13.2f}"
            for S in Ss
        )
        line += "  || "
        line += "  ".join(
            f"{next(r for r in rows if r['B']==B and r['S']==S)['ms_per_token']:>13.2f}"
            for S in Ss
        )
        print(line)


if __name__ == "__main__":
    main()
