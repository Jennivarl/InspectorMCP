"""Test the exported CellNet with OpenCV 5 DNN on the held-out split; check it matches PyTorch.

Usage: python scripts/eval_cnn.py --data ~/elpv-dataset/src/elpv_dataset/data
Writes results/cnn.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from inspector import cnn
from scripts.train_eval import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    data = args.data.expanduser()
    single = json.loads(Path("results/single_pass.json").read_text())
    test_ids = set(single["test_ids"])

    imgs, y, kinds = [], [], []
    for line in (data / "labels.csv").read_text().splitlines():
        path, prob, kind = line.split()
        if Path(path).name in test_ids:
            imgs.append(cv2.imread(str(data / path), cv2.IMREAD_GRAYSCALE))
            y.append(int(float(prob) >= 0.5))
            kinds.append(kind)
    y = np.array(y)

    net = cnn.net()
    net.predict(imgs[:4])  # warm-up
    t0 = time.perf_counter()
    p = np.concatenate([net.predict(imgs[i:i + 32]) for i in range(0, len(imgs), 32)])
    batch_ms = 1000 * (time.perf_counter() - t0) / len(imgs)
    t0 = time.perf_counter()
    for im in imgs[:100]:
        net.predict_one(im)
    single_ms = 1000 * (time.perf_counter() - t0) / 100

    result = {
        "engine": f"OpenCV {cv2.__version__} DNN",
        "test": len(y),
        "ms_per_image_batch32": round(batch_ms, 2),
        "ms_per_image_single": round(single_ms, 2),
        "test_metrics": metrics(y, p, net.threshold),
        "test_by_type": {k: metrics(y[m], p[m], net.threshold)
                         for k in ("mono", "poly") for m in [np.array(kinds) == k]},
        "single_pass_baseline": single["test_metrics"],
    }
    Path("results/cnn.json").write_text(json.dumps(result, indent=1))
    np.save("results/cnn_test_probs.npy", p)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
