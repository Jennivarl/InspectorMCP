"""CellNet inference with the OpenCV 5 DNN module (runs on COOL, no PyTorch needed)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import vision as v

MODEL_PATH = Path(__file__).with_name("cellnet.onnx")


class CellNet:
    def __init__(self, path: Path = MODEL_PATH):
        meta = json.loads(path.with_suffix(".json").read_text())
        self.size = meta["size"]
        self.threshold = meta["threshold"]
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def _blob(self, cells: list[np.ndarray]) -> np.ndarray:
        imgs = [cv2.resize(v.to_gray(c), (self.size, self.size), interpolation=cv2.INTER_AREA) for c in cells]
        return cv2.dnn.blobFromImages(imgs, scalefactor=1 / 255.0)

    def predict(self, cells: list[np.ndarray]) -> np.ndarray:
        """P(defect) for each cell image."""
        self.net.setInput(self._blob(cells))
        return np.asarray(self.net.forward()).reshape(-1)

    def predict_one(self, cell: np.ndarray) -> float:
        return float(self.predict([cell])[0])

    def heatmap(self, cell: np.ndarray, grid: int = 3) -> np.ndarray:
        """Where is the defect? Grey out each patch in turn; a big drop in P(defect) marks the spot."""
        g = v.normalize_cell(cell)
        base = self.predict_one(g)
        h, w = g.shape
        fill = int(np.median(g))
        variants, cells_ = [], []
        for r in range(grid):
            for c in range(grid):
                y0, y1 = r * h // grid, (r + 1) * h // grid
                x0, x1 = c * w // grid, (c + 1) * w // grid
                occ = g.copy()
                occ[y0:y1, x0:x1] = fill
                variants.append(occ)
                cells_.append((r, c))
        drops = base - self.predict(variants)
        heat = np.zeros((grid, grid), np.float32)
        for (r, c), d in zip(cells_, drops):
            heat[r, c] = d
        return heat


_NET: CellNet | None = None


def net() -> CellNet:
    global _NET
    if _NET is None:
        _NET = CellNet()
    return _NET
