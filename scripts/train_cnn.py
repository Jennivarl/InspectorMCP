"""Fine-tune a small CNN (MobileNetV3-Small) on ELPV and export it to ONNX for OpenCV 5 DNN.

Training only - the live app runs the exported model with cv2.dnn (COOL), no PyTorch.
Uses the same held-out test split as scripts/train_eval.py.

Usage (training venv with COOL cv2 on the path):
  python scripts/train_cnn.py --data ~/elpv-dataset/src/elpv_dataset/data --epochs 12
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

SIZE = 160
torch.manual_seed(7)
np.random.seed(7)


class CellNet(nn.Module):
    """Takes a 1-channel image scaled to 0..1 (what cv2.dnn.blobFromImage gives) and returns P(defect)."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        return torch.sigmoid(self.net(x))


def preprocess(g: np.ndarray) -> np.ndarray:
    """Identical to the runtime path in inspector/cnn.py."""
    return cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    if torch.rand(1) < 0.5:
        x = x.flip(-2)
    gain = 1 + 0.25 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
    bias = 0.1 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
    return (x * gain + bias).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("inspector/cellnet.onnx"))
    args = ap.parse_args()
    data = args.data.expanduser()
    torch.set_num_threads(max(torch.get_num_threads(), 2))

    test_ids = set(json.loads(Path("results/single_pass.json").read_text())["test_ids"])
    X, P, is_test = [], [], []
    for line in (data / "labels.csv").read_text().splitlines():
        path, prob, _ = line.split()
        X.append(preprocess(cv2.imread(str(data / path), cv2.IMREAD_GRAYSCALE)))
        P.append(float(prob))
        is_test.append(Path(path).name in test_ids)
    X = torch.from_numpy(np.stack(X)).float().div(255).unsqueeze(1)
    P = torch.tensor(P)
    is_test = np.array(is_test)

    train_idx = np.where(~is_test)[0]
    rng = np.random.default_rng(7)
    rng.shuffle(train_idx)
    n_val = len(train_idx) // 10
    val_idx, fit_idx = train_idx[:n_val], train_idx[n_val:]

    model = CellNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * (len(fit_idx) // 32 + 1))
    y_fit = (P[fit_idx] >= 0.5).float()
    pos_w = (1 - y_fit.mean()) / y_fit.mean()

    def evaluate(idx):
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(X[idx[i:i + 64]]).squeeze(1) for i in range(0, len(idx), 64)])
        return out.numpy()

    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(fit_idx)
        t0, total = time.time(), 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            x = augment(X[b])
            target = P[b]  # soft labels: ELPV gives defect probability 0 / .33 / .67 / 1
            w = torch.where(target >= 0.5, pos_w, torch.tensor(1.0))
            prob = model(x).squeeze(1).clamp(1e-6, 1 - 1e-6)
            loss = (nn.functional.binary_cross_entropy(prob, target, reduction="none") * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        pv = evaluate(val_idx)
        yv = (P[val_idx] >= 0.5).numpy()
        acc = float(((pv >= 0.5) == yv).mean())
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(perm):.4f} val_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    pv = evaluate(val_idx)
    yv = (P[val_idx] >= 0.5).numpy()
    thr = max(np.linspace(0.2, 0.8, 61), key=lambda t: float(((pv >= t) == yv).mean()))
    torch.onnx.export(model, torch.zeros(1, 1, SIZE, SIZE), str(args.out), input_names=["image"],
                      output_names=["p_defect"], dynamic_axes={"image": {0: "n"}, "p_defect": {0: "n"}},
                      opset_version=17, dynamo=False)
    meta = {"size": SIZE, "threshold": float(thr), "val_acc": best, "epochs": args.epochs, "arch": "mobilenet_v3_small"}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("saved", args.out, meta, flush=True)


if __name__ == "__main__":
    main()
