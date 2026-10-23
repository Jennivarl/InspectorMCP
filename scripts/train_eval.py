"""Extract OpenCV features for every ELPV cell, train the scorer, report test results.

Usage: python scripts/train_eval.py --data ~/elpv-dataset/src/elpv_dataset/data
Writes inspector/model.json and results/single_pass.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from inspector import model as m
from inspector import vision as v


def load_labels(data: Path):
    rows = []
    for line in (data / "labels.csv").read_text().splitlines():
        path, prob, kind = line.split()
        rows.append((data / path, float(prob), kind))
    return rows


def metrics(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    order = np.argsort(-p)
    ys = y[order]
    auc = float(((np.cumsum(1 - ys)[ys == 1]).sum()) / max(ys.sum() * (1 - ys).sum(), 1))
    auc = 1 - auc  # fraction of (pos, neg) pairs ranked correctly
    return {
        "threshold": round(float(thr), 3),
        "accuracy": round((tp + tn) / len(y), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "auc": round(auc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    rows = load_labels(args.data.expanduser())
    feats, t0 = [], time.perf_counter()
    for path, _, _ in rows:
        feats.append(v.features(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)))
    secs = time.perf_counter() - t0

    X = np.stack([m.to_vector(f) for f in feats])
    prob = np.array([r[1] for r in rows])
    y = (prob >= 0.5).astype(int)  # standard ELPV split: >= 50% defect probability = defective

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(rows))
    test = np.zeros(len(rows), bool)
    for cls in (0, 1):  # stratified 25% test split
        c = idx[y[idx] == cls]
        test[c[: len(c) // 4]] = True

    model = m.train(X[~test], y[~test])
    p_train = m.predict_proba(model, X[~test])
    # pick the threshold with best F1 on the training part only
    thr = max(np.linspace(0.2, 0.8, 61), key=lambda t: metrics(y[~test], p_train, t)["f1"])
    model["threshold"] = float(thr)
    m.save(model)

    p_test = m.predict_proba(model, X[test])
    result = {
        "images": len(rows),
        "train": int((~test).sum()),
        "test": int(test.sum()),
        "feature_ms_per_image": round(1000 * secs / len(rows), 2),
        "test_metrics": metrics(y[test], p_test, thr),
        "test_by_type": {
            kind: metrics(y[test][k], p_test[k], thr)
            for kind in ("mono", "poly")
            for k in [np.array([r[2] == kind for r in rows])[test]]
        },
        "test_ids": [str(rows[i][0].name) for i in np.where(test)[0]],
    }
    (args.out / "single_pass.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v_ for k, v_ in result.items() if k != "test_ids"}, indent=1))


if __name__ == "__main__":
    main()
