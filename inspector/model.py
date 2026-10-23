"""Tiny defect scorer: logistic regression on the OpenCV features.

Plain numpy so it runs anywhere with no extra ML libraries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .vision import FEATURES

MODEL_PATH = Path(__file__).with_name("model.json")
# skewed counts (lengths, areas) behave better on a log scale
LOG_FEATURES = {"crack_length", "crack_count", "crack_max", "spot_count", "spot_area", "sharpness"}


def to_vector(feats: dict) -> np.ndarray:
    return np.array(
        [np.log1p(feats[k]) if k in LOG_FEATURES else feats[k] for k in FEATURES],
        dtype=np.float64,
    )


def train(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, steps: int = 4000, lr: float = 0.1) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    pos = y.mean()
    weights = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))  # balance classes
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * weights
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return {"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist(), "b": b}


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(model["mean"])) / np.array(model["std"])
    return 1 / (1 + np.exp(-(Z @ np.array(model["w"]) + model["b"])))


def save(model: dict, path: Path = MODEL_PATH) -> None:
    path.write_text(json.dumps(model, indent=1))


def load(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text())


def score(feats: dict, model: dict | None = None) -> float:
    model = model or load()
    return float(predict_proba(model, to_vector(feats)[None, :])[0])
