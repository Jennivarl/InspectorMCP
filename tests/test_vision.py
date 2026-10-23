"""Unit tests for the OpenCV 5 tools - synthetic images, no dataset needed."""

import cv2
import numpy as np

from inspector import vision as v


def synthetic_cell(crack: bool = False, dark: bool = False) -> np.ndarray:
    """Grey EL-like cell with three busbars, optionally a diagonal crack or a dark patch."""
    rng = np.random.default_rng(0)
    g = np.clip(rng.normal(150, 4, (300, 300)), 0, 255).astype(np.uint8)
    for y in (75, 150, 225):
        g[y - 2:y + 3, :] = 70
    if crack:
        cv2.line(g, (60, 40), (200, 260), 60, 2)
    if dark:
        cv2.circle(g, (90, 110), 45, 40, -1)
    return cv2.GaussianBlur(g, (3, 3), 0)


def test_flatten_keeps_shape_and_centres_on_128():
    g = synthetic_cell()
    flat = v.flatten(g)
    assert flat.shape == g.shape and flat.dtype == np.uint8
    assert 115 < float(np.median(flat)) < 140


def test_busbars_are_masked_not_reported_as_cracks():
    g = synthetic_cell()
    assert v.busbar_mask(g)[150, 150] == 255
    assert v.find_cracks(g)["cracks"] == []


def test_crack_is_found_and_located():
    res = v.find_cracks(synthetic_cell(crack=True))
    assert res["cracks"], "diagonal crack should be detected"
    x, y, w, h = res["cracks"][0]["box"]
    assert w > 40 and h > 40  # spans a good part of the line
    assert res["crack_length"] > 60


def test_dark_area_fraction():
    clean = v.find_dark_areas(synthetic_cell())["dark_fraction"]
    dark = v.find_dark_areas(synthetic_cell(dark=True))["dark_fraction"]
    assert clean < 0.01 < dark


def test_shared_prep_gives_same_features():
    g = synthetic_cell(crack=True)
    assert v.features(g) == v.features(g, prep=v.Prep(g))


def test_panel_round_trip():
    cells = [synthetic_cell(crack=(i == 5)) for i in range(24)]
    panel = v.make_panel(cells, 4, 6)
    boxes = v.split_panel(v.rectify_panel(panel), 4, 6)
    assert len(boxes) == 24
    x, y, w, h = boxes[5]["box"]
    assert abs(w - 300) < 15 and abs(h - 300) < 15


def test_line_check_prefers_real_lines():
    cracked = v.line_check(synthetic_cell(crack=True), [60, 40, 140, 220])
    clean = v.line_check(synthetic_cell(), [60, 40, 140, 220])
    assert cracked["longest_line"] > clean["longest_line"]
