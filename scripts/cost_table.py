"""Cost per million inspected cells, from measured throughput and on-demand prices.

Prices (us-east-1, Linux, on-demand; checked Oct 2026):
  c8g.large  $0.0798/h (Holori, data 2026-07-16)   COOL Graviton4 software fee $0.01/h (AWS Marketplace listing)
  c7i.large  $0.0892/h (Holori, data 2026-07-16)
Usage: python scripts/cost_table.py  -> results/cost.json
"""

import json
from pathlib import Path

RUNS = {
    "Graviton4 c8g.large + COOL (OpenCV 5.1)": (["cool_g4_a", "cool_g4_b"], 0.0798 + 0.01),
    "Graviton4 c8g.large + stock OpenCV 5.0 wheel": (["stock_g4_a", "stock_g4_b"], 0.0798),
    "x86 c7i.large (Xeon 8488C) + stock OpenCV 5.0 wheel": (["stock_x86_a", "stock_x86_b"], 0.0892),
}


def main():
    rows = []
    for name, (labels, price) in RUNS.items():
        runs = [json.loads(Path(f"results/bench_{l}.json").read_text()) for l in labels]
        cps = sum(r["cells_per_second"] for r in runs) / len(runs)
        ms = sum(r["agent_scan_ms_per_cell"] for r in runs) / len(runs)
        rows.append({
            "setup": name,
            "usd_per_hour": round(price, 4),
            "scan_ms_per_cell": round(ms, 2),
            "cells_per_second": round(cps, 1),
            "usd_per_million_cells": round(price / (cps * 3600) * 1e6, 3),
        })
    Path("results/cost.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['setup']:55s} {r['cells_per_second']:7.1f} cells/s  ${r['usd_per_million_cells']:.3f} per 1M cells")


if __name__ == "__main__":
    main()
