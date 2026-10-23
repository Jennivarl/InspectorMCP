"""MCP server: OpenCV 5 (COOL) inspection tools any agent can call.

Perception tools return numbers and image ids; action tools file tickets or
ask a human for approval. Run standalone with:  python -m inspector.server
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from mcp.server.mcpserver import Image, MCPServer

from . import cnn
from . import model as scorer
from . import vision as v
from .store import STORE

DATA_DIR = Path.home() / "elpv-dataset/src/elpv_dataset/data"
PASS_BELOW, FAIL_ABOVE = 0.25, 0.75  # between the two = "uncertain", look closer

mcp = MCPServer(
    name="inspector",
    title="InspectorMCP",
    description="OpenCV 5 (COOL on AWS Graviton) tools for inspecting solar cell EL images.",
    instructions=(
        "Inspect solar cells. Typical flow: split a panel into cells, scan_cell each one, "
        "look closer (inspect_region / enhance_image) when the verdict is 'uncertain', then act: "
        "pass, open_ticket for clear defects, or request_approval when unsure."
    ),
)

_MODEL = None


def _model() -> dict:
    global _MODEL
    if _MODEL is None:
        _MODEL = scorer.load()
    return _MODEL


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, round(1000 * (time.perf_counter() - t0), 2)


def _public(d: dict) -> dict:
    return {k: val for k, val in d.items() if not k.startswith("_")}


# ---------- inputs ----------

@mcp.tool()
def opencv_info() -> dict:
    """Which OpenCV build is doing the work (proves COOL / KleidiCV is active)."""
    info = cv2.getBuildInformation()
    keep = [l.strip() for l in info.splitlines() if any(k in l for k in ("Custom HAL", "Baseline", "Parallel framework"))]
    return {"opencv_version": cv2.__version__, "module": cv2.__file__, "build": keep}


@mcp.tool()
def make_demo_panel(rows: int = 4, cols: int = 6, defect_share: float = 0.25, seed: int = 0) -> dict:
    """Build a demo panel image from real dataset cells. Returns the panel image_id."""
    lines = (DATA_DIR / "labels.csv").read_text().splitlines()
    rows_ = [l.split() for l in lines]
    good = [r for r in rows_ if float(r[1]) == 0.0]
    bad = [r for r in rows_ if float(r[1]) == 1.0]
    rng = np.random.default_rng(seed)
    n = rows * cols
    n_bad = int(round(n * defect_share))
    picks = [bad[i] for i in rng.choice(len(bad), n_bad, replace=False)]
    picks += [good[i] for i in rng.choice(len(good), n - n_bad, replace=False)]
    rng.shuffle(picks)
    cells = [cv2.imread(str(DATA_DIR / p[0]), cv2.IMREAD_GRAYSCALE) for p in picks]
    panel = v.make_panel(cells, rows, cols)
    panel_id = STORE.put(panel, "panel", rows=rows, cols=cols, source="demo")
    STORE.meta[panel_id]["truth"] = [int(float(p[1]) >= 0.5) for p in picks]  # hidden from tools
    return {"image_id": panel_id, "rows": rows, "cols": cols, "size": list(panel.shape)}


@mcp.tool()
def load_dataset_cell(name: str) -> dict:
    """Load one cell from the public ELPV dataset, e.g. 'cell0006.png'."""
    path = DATA_DIR / "images" / Path(name).name
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"not found: {name}")
    return {"image_id": STORE.put(img, "cell", source=name)}


@mcp.tool()
def split_panel(image_id: str, rows: int, cols: int) -> dict:
    """Straighten a panel image and cut it into cells. Returns one image_id per cell."""
    panel = STORE.get(image_id)
    flat, ms1 = _timed(v.rectify_panel, panel)
    boxes, ms2 = _timed(v.split_panel, flat, rows, cols)
    truth = STORE.meta.get(image_id, {}).get("truth")
    cells = []
    for i, b in enumerate(boxes):
        x, y, w, h = b["box"]
        cid = STORE.put(v.normalize_cell(flat[y:y + h, x:x + w]), "cell", panel=image_id, row=b["row"], col=b["col"])
        if truth:
            STORE.truth[cid] = truth[i]
        cells.append({"image_id": cid, "row": b["row"], "col": b["col"]})
    return {"cells": cells, "ms": round(ms1 + ms2, 2)}


# ---------- perception ----------

@mcp.tool()
def check_quality(image_id: str) -> dict:
    """Sharpness / brightness / contrast. Low values mean the image needs enhancing."""
    q, ms = _timed(v.quality, STORE.get(image_id))
    return {**q, "ms": ms}


@mcp.tool()
def enhance_image(image_id: str, method: str = "clahe") -> dict:
    """Make faint marks easier to see. method: clahe | denoise | sharpen | normalize. Returns a new image_id."""
    out, ms = _timed(v.enhance, STORE.get(image_id), method)
    new_id = STORE.put(out, "cell", parent=image_id, enhanced=method)
    if image_id in STORE.truth:
        STORE.truth[new_id] = STORE.truth[image_id]
    return {"image_id": new_id, "ms": ms}


def _verdict(p: float) -> str:
    return "pass" if p < PASS_BELOW else "fail" if p > FAIL_ABOVE else "uncertain"


@mcp.tool()
def scan_cell(image_id: str, sensitivity: float = 4.0) -> dict:
    """Full check of one cell: defect score (0-1) from the CellNet model (OpenCV DNN),
    plus where the marks are (cracks, spots, dark areas) from classic OpenCV.

    verdict: 'pass' (score < 0.25), 'fail' (> 0.75) or 'uncertain' (look closer).
    Lower sensitivity finds fainter cracks (and more false alarms).
    """
    g = STORE.get(image_id)
    t0 = time.perf_counter()
    score = cnn.net().predict_one(g)
    prep = v.Prep(g)
    dark = v.find_dark_areas(g, prep=prep)
    base_cracks = v.find_cracks(g, prep=prep)
    feats = v.features(g, prep=prep, cracks=base_cracks, dark=dark)
    cracks = base_cracks if sensitivity == 4.0 else v.find_cracks(g, sensitivity=sensitivity, prep=prep)
    rule_score = scorer.score(feats, _model())
    ms = round(1000 * (time.perf_counter() - t0), 2)
    return {
        "image_id": image_id,
        "score": round(score, 3),
        "verdict": _verdict(score),
        "rule_score": round(rule_score, 3),
        "cracks": [_public(c) for c in cracks["cracks"][:5]],
        "spots": [_public(s) for s in cracks["spots"][:5]],
        "dark_areas": dark["areas"][:3],
        "dark_fraction": dark["dark_fraction"],
        "grain": feats["grain"],
        "ms": ms,
    }


@mcp.tool()
def second_look(image_id: str) -> dict:
    """Re-score the cell from several views (mirrored, flipped, contrast-boosted, denoised).

    A steady score across views is trustworthy; a big spread means stay careful.
    """
    g = v.normalize_cell(STORE.get(image_id))
    views = {
        "original": g, "mirror": cv2.flip(g, 1), "flip": cv2.flip(g, 0), "rotate180": cv2.flip(g, -1),
        "clahe": v.enhance(g, "clahe"), "denoise": v.enhance(g, "denoise"),
    }
    probs, ms = _timed(cnn.net().predict, list(views.values()))
    mean = float(np.mean(probs))
    return {
        "image_id": image_id,
        "scores": {k: round(float(p), 3) for k, p in zip(views, probs)},
        "mean": round(mean, 3),
        "spread": round(float(np.max(probs) - np.min(probs)), 3),
        "verdict": _verdict(mean),
        "ms": ms,
    }


@mcp.tool()
def locate_defect(image_id: str, grid: int = 3) -> dict:
    """Find where the model sees the defect: hide each patch in turn and watch the score drop.

    Returns a heat grid and the box of the hottest patch (feed it to inspect_region).
    """
    g = v.normalize_cell(STORE.get(image_id))
    heat, ms = _timed(cnn.net().heatmap, g, grid)
    r, c = np.unravel_index(int(np.argmax(heat)), heat.shape)
    h, w = g.shape
    box = [int(c * w // grid), int(r * h // grid), int(w // grid), int(h // grid)]
    return {"image_id": image_id, "heat": np.round(heat, 3).tolist(), "hot_box": box,
            "hot_drop": round(float(heat[r, c]), 3), "ms": ms}


@mcp.tool()
def inspect_region(image_id: str, x: int, y: int, w: int, h: int) -> list:
    """Zoom into one region (from scan_cell boxes) and test if it is a real crack.

    Returns line measurements plus the zoomed picture so a vision model can look too.
    Real cracks: longest_line > ~25 px, straightness > ~0.5, contrast > ~8.
    """
    g = STORE.get(image_id)
    check, ms = _timed(v.line_check, g, [x, y, w, h])
    crop = v.enhance(v.zoom(g, [x, y, w, h], scale=3), "clahe")
    return [json.dumps({**check, "ms": ms}), Image(data=_png(crop), format="png")]


@mcp.tool()
def view_cell(image_id: str) -> list:
    """Picture of the cell with findings drawn on it (red = crack, yellow = spot, blue = dark area)."""
    g = STORE.get(image_id)
    over = v.draw_findings(g, v.find_cracks(g), v.find_dark_areas(g))
    return [Image(data=_png(over), format="png")]


# ---------- actions ----------

@mcp.tool()
def mark_pass(image_id: str, reason: str) -> dict:
    """Accept a cell as healthy."""
    return STORE.set_verdict(image_id, "pass", reason)


@mcp.tool()
def open_ticket(image_id: str, defect: str, severity: str, evidence: str) -> dict:
    """Send a cell for repair/replacement. severity: low | medium | high."""
    meta = STORE.meta.get(image_id, {})
    return STORE.add_ticket(image_id=image_id, panel=meta.get("panel"), row=meta.get("row"), col=meta.get("col"),
                            defect=defect, severity=severity, evidence=evidence)


@mcp.tool()
def request_approval(image_id: str, proposed_action: str, reason: str) -> dict:
    """Ask a human to confirm before acting when the evidence is not clear."""
    meta = STORE.meta.get(image_id, {})
    return STORE.request_approval(image_id=image_id, panel=meta.get("panel"), row=meta.get("row"),
                                  col=meta.get("col"), proposed_action=proposed_action, reason=reason)


@mcp.tool()
def approval_status(approval_id: str) -> dict:
    """Has the human answered an approval request yet?"""
    return STORE.approvals[approval_id]


if __name__ == "__main__":
    mcp.run("streamable-http", host="127.0.0.1", port=8001)
