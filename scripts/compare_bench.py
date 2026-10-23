"""Side-by-side table of two benchmark runs.

Usage: python scripts/compare_bench.py results/bench_stock_graviton4.json results/bench_cool_graviton4.json
"""

import json
import sys


def main(base_path: str, new_path: str):
    b, a = json.load(open(base_path)), json.load(open(new_path))
    print(f"baseline: {b['label']}  OpenCV {b['opencv']}  {b['hal']}  {b['baseline_simd']}")
    print(f"compared: {a['label']}  OpenCV {a['opencv']}  {a['hal']}  {a['baseline_simd']}")
    print(f"{'stage':30s} {'base ms':>9s} {'new ms':>9s} {'speedup':>8s}")
    for k in a["stage_ms"]:
        if k not in b["stage_ms"]:
            continue
        print(f"{k:30s} {b['stage_ms'][k]:9.3f} {a['stage_ms'][k]:9.3f} {b['stage_ms'][k] / a['stage_ms'][k]:7.2f}x")
    for k in ("agent_scan_ms_per_cell", "panel_24_cells_ms"):
        print(f"{k:30s} {b[k]:9.2f} {a[k]:9.2f} {b[k] / a[k]:7.2f}x")
    print(f"{'cells_per_second':30s} {b['cells_per_second']:9.1f} {a['cells_per_second']:9.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
