"""Time single OpenCV operations used by InspectorMCP (whichever build is on the path).

Usage: python scripts/op_bench.py --label cool  -> results/ops_<label>.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def bench(fn, n=200, repeat=5):
    fn()
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        runs.append(1e6 * (time.perf_counter() - t0) / n)
    return round(statistics.median(runs), 2)  # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    g = cv2.GaussianBlur(rng.integers(40, 110, (300, 300), dtype=np.uint8), (3, 3), 0)
    big = cv2.resize(g, (1842, 1230))
    f32 = g.astype(np.float32)
    e15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    e7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mask = (g > 80).astype(np.uint8) * 255

    ops = {
        "medianBlur_k31": lambda: cv2.medianBlur(g, 31),
        "medianBlur_k61": lambda: cv2.medianBlur(g, 61),
        "medianBlur_k3": lambda: cv2.medianBlur(g, 3),
        "GaussianBlur_7x7": lambda: cv2.GaussianBlur(g, (7, 7), 0),
        "GaussianBlur_sigma3_panel": lambda: cv2.GaussianBlur(big, (0, 0), 3),
        "blur_box_31": lambda: cv2.blur(g, (31, 31)),
        "resize_area_down_160": lambda: cv2.resize(g, (160, 160), interpolation=cv2.INTER_AREA),
        "resize_linear_down_75": lambda: cv2.resize(g, (75, 75), interpolation=cv2.INTER_LINEAR),
        "resize_linear_up_300": lambda: cv2.resize(cv2.resize(g, (75, 75)), (300, 300), interpolation=cv2.INTER_LINEAR),
        "resize_cubic_up_3x": lambda: cv2.resize(g[:80, :80], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
        "morph_blackhat_e15": lambda: cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, e15),
        "morph_open_e7": lambda: cv2.morphologyEx(g, cv2.MORPH_OPEN, e7),
        "erode_rect_3": lambda: cv2.erode(g, np.ones((3, 3), np.uint8)),
        "divide_scale": lambda: cv2.divide(g, g, scale=128),
        "threshold_otsu": lambda: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU),
        "clahe": lambda: clahe.apply(g),
        "laplacian_64f": lambda: cv2.Laplacian(g, cv2.CV_64F),
        "meanStdDev": lambda: cv2.meanStdDev(g),
        "connectedComponentsWithStats": lambda: cv2.connectedComponentsWithStats(mask, connectivity=8),
        "findContours": lambda: cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE),
        "flip": lambda: cv2.flip(g, 1),
        "cvtColor_gray2bgr": lambda: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
        "convertTo_f32": lambda: g.astype(np.float32),
        "addWeighted": lambda: cv2.addWeighted(g, 1.6, g, -0.6, 0),
        "normalize_minmax": lambda: cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX),
        "warpPerspective_panel": lambda: cv2.warpPerspective(big, np.eye(3), (1842, 1230)),
        "imencode_png": lambda: cv2.imencode(".png", g),
        "sobel_f32": lambda: cv2.Sobel(f32, cv2.CV_32F, 1, 0),
    }
    res = {"label": args.label, "opencv": cv2.__version__, "threads": cv2.getNumThreads(),
           "us": {k: bench(fn) for k, fn in ops.items()}}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/ops_{args.label}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
