"""Time the real InspectorMCP workload with whichever OpenCV build is on the path.

Run it once under COOL and once under a stock OpenCV wheel (same machine, or an x86
machine) and compare. Stages are exactly what the agent calls per cell.

Usage: python scripts/benchmark.py --data ~/elpv-dataset/src/elpv_dataset/data --label cool
Writes results/bench_<label>.json
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from inspector import cnn
from inspector import vision as v


def timeit(fn, items, repeat=3):
    """Median over repeats of the mean ms per item."""
    fn(items[0])  # warm-up
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        runs.append(1000 * (time.perf_counter() - t0) / len(items))
    return round(statistics.median(runs), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    data = args.data.expanduser()

    names = sorted((data / "images").glob("*.png"))[: args.n]
    cells = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in names]
    panels = [v.make_panel(cells[i:i + 24], 4, 6) for i in range(0, len(cells) - 23, 24)]
    net = cnn.net()

    stages = {
        "1_split_panel_24_cells": timeit(lambda p: v.split_panel(v.rectify_panel(p), 4, 6), panels),
        "2_quality": timeit(v.quality, cells),
        "3_enhance_clahe": timeit(lambda g: v.enhance(g, "clahe"), cells),
        "4_find_cracks": timeit(v.find_cracks, cells),
        "5_find_dark_areas": timeit(v.find_dark_areas, cells),
        "6_features_full": timeit(v.features, cells),
        "7_cellnet_dnn_single": timeit(net.predict_one, cells),
        "8_second_look_6_views": timeit(lambda g: net.predict([g, cv2.flip(g, 1), cv2.flip(g, 0), cv2.flip(g, -1),
                                                              v.enhance(g, "clahe"), v.enhance(g, "denoise")]), cells[:60]),
        "9_locate_defect_3x3": timeit(lambda g: net.heatmap(g, 3), cells[:60]),
        "10_zoom_line_check": timeit(lambda g: v.line_check(g, [100, 100, 60, 60]), cells),
    }
    def scan(g):
        """Exactly what the scan_cell tool does (shared Prep) plus the quality check."""
        v.quality(g)
        net.predict_one(g)
        prep = v.Prep(g)
        dark = v.find_dark_areas(g, prep=prep)
        cracks = v.find_cracks(g, prep=prep)
        v.features(g, prep=prep, cracks=cracks, dark=dark)

    stages["11_scan_cell_tool"] = timeit(scan, cells)
    per_cell = stages["11_scan_cell_tool"]

    # throughput: full scan of a 24-cell panel end to end
    def panel_scan(p):
        flat = v.rectify_panel(p)
        for b in v.split_panel(flat, 4, 6):
            x, y, w, h = b["box"]
            scan(v.normalize_cell(flat[y:y + h, x:x + w]))
    panel_ms = timeit(panel_scan, panels, repeat=3)

    info = cv2.getBuildInformation()
    result = {
        "label": args.label,
        "opencv": cv2.__version__,
        "opencv_module": cv2.__file__,
        "hal": next((l.strip() for l in info.splitlines() if "Custom HAL" in l), "none"),
        "baseline_simd": next((l.strip() for l in info.splitlines() if "Baseline:" in l), ""),
        "machine": platform.machine(),
        "cpu_threads": cv2.getNumThreads(),
        "cells": len(cells),
        "stage_ms": stages,
        "agent_scan_ms_per_cell": round(per_cell, 3),
        "panel_24_cells_ms": panel_ms,
        "cells_per_second": round(24 * 1000 / panel_ms, 1),
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/bench_{args.label}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
