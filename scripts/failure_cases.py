"""Montage of CellNet's worst mistakes on the held-out test set (evidence of limitations).

Usage: python -m scripts.failure_cases --data ~/elpv-dataset/src/elpv_dataset/data
Writes docs/failure_cases.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from inspector import cnn

TILE = 200


def label(img, text, color):
    out = cv2.cvtColor(cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, TILE - 26), (TILE, TILE), (20, 20, 20), -1)
    cv2.putText(out, text, (6, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    data = args.data.expanduser()
    test_ids = json.loads(Path("results/single_pass.json").read_text())["test_ids"]
    probs = {}
    for line in (data / "labels.csv").read_text().splitlines():
        path, p, _ = line.split()
        probs[Path(path).name] = float(p)
    net = cnn.net()
    rows = []
    for name in test_ids:
        g = cv2.imread(str(data / "images" / name), cv2.IMREAD_GRAYSCALE)
        rows.append((name, int(probs[name] >= 0.5), net.predict_one(g), g))
    missed = sorted([r for r in rows if r[1] == 1], key=lambda r: r[2])[: args.n]          # defects scored lowest
    false_alarm = sorted([r for r in rows if r[1] == 0], key=lambda r: -r[2])[: args.n]    # healthy scored highest
    top = np.hstack([label(g, f"missed  p={p:.2f}", (110, 110, 255)) for _, _, p, g in missed])
    bottom = np.hstack([label(g, f"false alarm  p={p:.2f}", (90, 200, 255)) for _, _, p, g in false_alarm])
    Path("docs").mkdir(exist_ok=True)
    cv2.imwrite("docs/failure_cases.png", np.vstack([top, np.full((8, top.shape[1], 3), 255, np.uint8), bottom]))
    print("missed:", [r[0] for r in missed])
    print("false alarms:", [r[0] for r in false_alarm])


if __name__ == "__main__":
    main()
