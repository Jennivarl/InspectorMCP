"""OpenCV 5 inspection tools for electroluminescence (EL) images of solar cells.

Everything here is classic imgproc (resize, blur, morphology, threshold,
contours, warps) - the operations COOL accelerates on AWS Graviton.
Images are uint8 grayscale numpy arrays unless noted.
"""

from __future__ import annotations

import cv2
import numpy as np

CELL = 300  # working size of one cell, in pixels


# ---------- basics ----------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize_cell(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    if g.shape != (CELL, CELL):
        g = cv2.resize(g, (CELL, CELL), interpolation=cv2.INTER_AREA)
    return g


def quality(img: np.ndarray) -> dict:
    """Is this image usable? Sharpness, brightness, contrast, clipping."""
    g = to_gray(img)
    mean, std = cv2.meanStdDev(g)
    return {
        "sharpness": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2),
        "brightness": round(float(mean[0][0]), 2),
        "contrast": round(float(std[0][0]), 2),
        "clipped_dark": round(float((g <= 5).mean()), 4),
        "clipped_bright": round(float((g >= 250).mean()), 4),
    }


def enhance(img: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Make faint defects easier to see."""
    g = to_gray(img)
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    if method == "denoise":
        return cv2.medianBlur(cv2.GaussianBlur(g, (3, 3), 0), 3)
    if method == "sharpen":
        soft = cv2.GaussianBlur(g, (0, 0), 2.0)
        return cv2.addWeighted(g, 1.6, soft, -0.6, 0)
    if method == "normalize":
        return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    raise ValueError(f"unknown enhance method: {method}")


# ---------- cell structure ----------

def background(g: np.ndarray, k: int = 31) -> np.ndarray:
    """Slow brightness changes across the cell.

    A large median filter is robust to thin dark lines but slow (and one of the few
    ops COOL does not speed up), so we take the median on a 4x smaller copy and
    scale it back: same robustness, ~16x fewer pixels, and the resizes are cheap.
    """
    h, w = g.shape
    small = cv2.resize(g, (max(w // 4, 3), max(h // 4, 3)), interpolation=cv2.INTER_AREA)
    k_small = max(3, (k // 4) | 1)
    return cv2.resize(cv2.medianBlur(small, k_small), (w, h), interpolation=cv2.INTER_LINEAR)


def flatten(g: np.ndarray, k: int = 31) -> np.ndarray:
    """Remove slow brightness changes so local dark marks stand out (~128 = normal)."""
    return cv2.divide(g, np.maximum(background(g, k), 1), scale=128)


def cell_mask(g: np.ndarray, margin: int = 8) -> np.ndarray:
    """Pixels that belong to the cell (drops dark corners and the outer frame)."""
    blur = cv2.GaussianBlur(g, (7, 7), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # closing re-joins the strips that thin dark lines (busbars, cracks) cut out of the mask
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    # Otsu also drops large dark (inactive) areas - the very defects we want to measure -
    # so keep the convex outline of the cell: corner cuts stay out, interior holes come back in
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        m = np.zeros_like(m)
        cv2.fillConvexPoly(m, hull, 255)
    frame = np.zeros_like(m)
    h, w = g.shape
    frame[margin:h - margin, margin:w - margin] = 255
    return cv2.bitwise_and(m, frame)


def busbar_mask(g: np.ndarray, pad: int = 4) -> np.ndarray:
    """Busbars: straight dark lines running the full width (or height) of the cell."""
    flat = flatten(g, 61).astype(np.float32)
    h, w = g.shape
    mask = np.zeros_like(g)
    for axis in (1, 0):  # 1 -> row profile (horizontal bars), 0 -> column profile
        # median, not mean: a busbar is dark across the whole width, a dark patch is not
        prof = np.median(flat[:, w // 6: w - w // 6], axis=1) if axis == 1 else np.median(flat[h // 6: h - h // 6, :], axis=0)
        med = np.median(prof)
        mad = np.median(np.abs(prof - med)) + 1e-3
        hits = np.where(prof < med - max(6 * mad, 6))[0]
        for i in hits:
            lo, hi = max(i - pad, 0), i + pad + 1
            if axis == 1:
                mask[lo:hi, :] = 255
            else:
                mask[:, lo:hi] = 255
    return mask


class Prep:
    """Per-cell work shared by every finder, computed once (was 3-4x per cell)."""

    def __init__(self, img: np.ndarray):
        self.g = normalize_cell(img)
        self.flat = flatten(self.g)
        self.roi = cv2.bitwise_and(cell_mask(self.g), cv2.bitwise_not(busbar_mask(self.g)))


def _prep(img: np.ndarray, prep: Prep | None) -> Prep:
    return prep if prep is not None else Prep(img)


# ---------- defect finders ----------

def _components(mask: np.ndarray, min_area: int) -> list[dict]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        out.append({"box": [x, y, w, h], "area": area})
    return out


def find_cracks(g: np.ndarray, kernel: int = 15, sensitivity: float = 4.0, prep: Prep | None = None) -> dict:
    """Thin dark lines that are not busbars: candidate cracks.

    Black-hat morphology lifts thin dark structures; long, thin components are
    kept as cracks, short fat ones as spots.
    """
    p = _prep(g, prep)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bh = cv2.morphologyEx(p.flat, cv2.MORPH_BLACKHAT, se)

    roi = p.roi
    vals = bh[roi > 0].astype(np.float32)
    if vals.size == 0:
        return {"cracks": [], "spots": [], "crack_length": 0.0, "blackhat_mean": 0.0, "blackhat_p99": 0.0}
    thr = max(12.0, float(vals.mean() + sensitivity * vals.std()))
    mask = ((bh > thr) & (roi > 0)).astype(np.uint8) * 255

    cracks, spots = [], []
    for c in _components(mask, min_area=12):
        x, y, w, h = c["box"]
        sub = mask[y:y + h, x:x + w]
        pts = cv2.findNonZero(sub)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        length = float(max(rw, rh))
        thickness = c["area"] / max(length, 1.0)
        strength = float(bh[y:y + h, x:x + w][sub > 0].mean())
        c.update(length=round(length, 1), thickness=round(thickness, 2), strength=round(strength, 1))
        if length >= 18 and length / max(thickness, 1.0) >= 4:
            cracks.append(c)
        elif c["area"] >= 30:
            spots.append(c)
    cracks.sort(key=lambda c: -c["length"])
    spots.sort(key=lambda c: -c["area"])
    return {
        "cracks": cracks,
        "spots": spots,
        "crack_length": round(sum(c["length"] for c in cracks), 1),
        "blackhat_mean": round(float(vals.mean()), 2),
        "blackhat_p99": round(float(np.percentile(vals, 99)), 2),
        "_mask": mask,
    }


def find_dark_areas(g: np.ndarray, ratio: float = 0.6, prep: Prep | None = None) -> dict:
    """Large dark patches (inactive or disconnected cell areas)."""
    p = _prep(g, prep)
    g, roi = p.g, p.roi
    inside = g[roi > 0]
    if inside.size == 0:
        return {"dark_fraction": 0.0, "areas": []}
    med = float(np.median(inside))
    smooth = cv2.GaussianBlur(g, (9, 9), 0)
    mask = ((smooth < ratio * med) & (roi > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    areas = sorted(_components(mask, min_area=60), key=lambda a: -a["area"])
    return {
        "dark_fraction": round(float((mask > 0).sum()) / float((roi > 0).sum()), 4),
        "areas": areas,
        "_mask": mask,
    }


def texture(g: np.ndarray, prep: Prep | None = None) -> dict:
    """How grainy the cell is (polycrystalline cells are naturally grainy)."""
    p = _prep(g, prep)
    g, roi = p.g, p.roi
    vals = p.flat[roi > 0].astype(np.float32)
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F)
    return {
        "grain": round(float(vals.std()) if vals.size else 0.0, 2),
        "edge_energy": round(float(np.abs(lap[roi > 0]).mean()) if vals.size else 0.0, 2),
    }


FEATURES = [
    "crack_length", "crack_count", "crack_max", "crack_strength",
    "spot_count", "spot_area", "dark_fraction", "dark_soft",
    "blackhat_mean", "blackhat_p99", "grain", "edge_energy",
    "sharpness", "brightness", "contrast",
]


def features(img: np.ndarray, prep: Prep | None = None, cracks: dict | None = None,
             dark: dict | None = None) -> dict:
    """One number per check, for the scoring model (reuses finder results when given)."""
    p = _prep(img, prep)
    g = p.g
    c = cracks if cracks is not None else find_cracks(g, prep=p)
    d = dark if dark is not None else find_dark_areas(g, prep=p)
    soft = find_dark_areas(g, ratio=0.8, prep=p)
    t = texture(g, prep=p)
    q = quality(g)
    return {
        "crack_length": c["crack_length"],
        "crack_count": len(c["cracks"]),
        "crack_max": c["cracks"][0]["length"] if c["cracks"] else 0.0,
        "crack_strength": max((k["strength"] for k in c["cracks"]), default=0.0),
        "spot_count": len(c["spots"]),
        "spot_area": float(sum(s["area"] for s in c["spots"])),
        "dark_fraction": d["dark_fraction"],
        "dark_soft": soft["dark_fraction"],
        "blackhat_mean": c["blackhat_mean"],
        "blackhat_p99": c["blackhat_p99"],
        "grain": t["grain"],
        "edge_energy": t["edge_energy"],
        "sharpness": q["sharpness"],
        "brightness": q["brightness"],
        "contrast": q["contrast"],
    }


# ---------- active perception ----------

def zoom(g: np.ndarray, box: list[int], scale: int = 3, pad: int = 12) -> np.ndarray:
    """Crop a region (with some context) and upscale it for a closer look."""
    g = to_gray(g)
    x, y, w, h = box
    H, W = g.shape
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    crop = g[y0:y1, x0:x1]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def line_check(g: np.ndarray, box: list[int]) -> dict:
    """Look closer at one region: is the dark mark a real straight-ish crack?

    Real cracks are long, fairly straight and clearly darker than their
    surroundings; grain boundaries in poly cells are short and wiggly.
    """
    z = enhance(zoom(g, box, scale=3), "denoise")
    flat = flatten(z, 61)
    bh = cv2.morphologyEx(flat, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=30, minLineLength=max(z.shape) // 4, maxLineGap=8)
    # OpenCV 5 returns (N, 4) / (N, 2); OpenCV 4 adds a middle axis - reshape covers both
    segs = np.empty((0, 4)) if lines is None else np.asarray(lines).reshape(-1, 4).astype(np.float64)
    longest = float(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).max()) if len(segs) else 0.0
    pts = cv2.findNonZero(m)
    straightness = 0.0
    if pts is not None and len(pts) >= 10:
        xy = np.asarray(pts).reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.abs((xy[:, 0] - x0) * vy - (xy[:, 1] - y0) * vx)
        straightness = float(1.0 / (1.0 + d.mean() / 3.0))
    inside = z[m > 0]
    outside = z[m == 0]
    contrast = float(outside.mean() - inside.mean()) if inside.size and outside.size else 0.0
    return {
        "longest_line": round(longest / 3.0, 1),  # back in cell pixels
        "line_count": int(len(segs)),
        "straightness": round(straightness, 3),
        "contrast": round(contrast, 1),
    }


# ---------- panels (many cells) ----------

def rectify_panel(img: np.ndarray) -> np.ndarray:
    """Find the panel outline in a photo and warp it flat (perspective correction)."""
    g = to_gray(img)
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return g
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.3 * g.size:
        return g
    quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(quad) != 4:
        return g
    pts = quad.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    src = np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(g, cv2.getPerspectiveTransform(src, dst), (w, h))


def _gaps(profile: np.ndarray, expected: int) -> list[int]:
    """Cut points between cells from a brightness profile (gaps are dark)."""
    n = len(profile)
    step = n / expected
    cuts = [0]
    for i in range(1, expected):
        lo, hi = int((i - 0.3) * step), int((i + 0.3) * step)
        cuts.append(lo + int(np.argmin(profile[lo:hi])))
    cuts.append(n)
    return cuts


def split_panel(img: np.ndarray, rows: int, cols: int) -> list[dict]:
    """Split a (rectified) panel image into its cells."""
    g = to_gray(img)
    smooth = cv2.GaussianBlur(g, (0, 0), 3).astype(np.float32)
    ys = _gaps(smooth.mean(axis=1), rows)
    xs = _gaps(smooth.mean(axis=0), cols)
    cells = []
    for r in range(rows):
        for c in range(cols):
            x0, x1, y0, y1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
            cells.append({"row": r, "col": c, "box": [x0, y0, x1 - x0, y1 - y0]})
    return cells


def make_panel(cells: list[np.ndarray], rows: int, cols: int, gap: int = 6) -> np.ndarray:
    """Tile single-cell images into one panel image (for demos and tests)."""
    tile = CELL + gap
    panel = np.full((rows * tile + gap, cols * tile + gap), 20, np.uint8)
    for i, cell in enumerate(cells[: rows * cols]):
        r, c = divmod(i, cols)
        y, x = gap + r * tile, gap + c * tile
        panel[y:y + CELL, x:x + CELL] = normalize_cell(cell)
    return panel


# ---------- drawing ----------

def draw_findings(g: np.ndarray, cracks: dict, dark: dict) -> np.ndarray:
    """Color overlay: red = cracks, yellow = spots, blue = dark areas."""
    g = normalize_cell(g)
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if "_mask" in dark:
        out[dark["_mask"] > 0] = (0.5 * out[dark["_mask"] > 0] + [127, 40, 0]).astype(np.uint8)
    for c in cracks.get("cracks", []):
        x, y, w, h = c["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for s in cracks.get("spots", []):
        x, y, w, h = s["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 255), 1)
    return out
