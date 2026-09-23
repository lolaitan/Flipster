import io

import cv2
import numpy as np
from PIL import Image

from flipster.preprocess import (
    estimate_paper_shift,
    ink_from_drawing,
    ink_from_scan,
    load_rgb,
    prepare_frames,
    remove_hole_punches,
    soft_field,
)
from synth import ruled_page


def test_ink_ignores_rules_and_margin():
    page = ruled_page(400, 300)
    cv2.line(page, (100, 50), (220, 300), (40, 40, 45), 3)
    ink = ink_from_scan(page)
    assert ink[175, 160] > 0.8 or ink[:, 150:170].max() > 0.8  # on the stroke
    rules = ink[::18, 60:90]  # blue rule rows, away from the stroke
    assert rules.max() < 0.1
    assert ink[:, 38:42].max() < 0.1  # red margin


def test_hole_punches_are_removed():
    page = ruled_page(600, 460)
    cv2.circle(page, (22, 300), 12, (90, 90, 90), -1)  # binder hole near the edge
    cv2.line(page, (150, 100), (300, 400), (30, 30, 30), 3)
    ink = remove_hole_punches(ink_from_scan(page))
    assert ink[290:310, 12:32].max() < 0.05
    assert ink[240:260, 170:230].max() > 0.8


def test_paper_shift_is_recovered():
    a = ruled_page(500, 380, margin_x=60)
    b = ruled_page(500, 380, margin_x=60, offset=(7, 3))
    dx, dy = estimate_paper_shift(a, b)
    assert (dx, dy) == (7.0, 3.0)
    blank = np.full_like(a, 250)
    assert estimate_paper_shift(blank, blank) == (0.0, 0.0)


def test_prepare_frames_aligns_pages():
    a = ruled_page(500, 380, margin_x=60)
    b = ruled_page(500, 380, margin_x=60, offset=(7, 3))
    frames = prepare_frames([a, b], source="scan", working_height=500)
    assert frames[1].transform[0, 2] == 7.0 and frames[1].transform[1, 2] == 3.0
    col = lambda rgb: int(np.argmax((rgb[..., 0] - rgb[..., 2])[100:400].mean(0)))  # noqa: E731
    assert col(frames[0].rgb) == col(frames[1].rgb)


def test_soft_field_widens_strokes():
    ink = np.zeros((50, 50), np.float32)
    ink[25, 5:45] = 1.0
    f = soft_field(ink, sigma=4.0)
    assert f[25, 25] == 1.0
    assert 0.2 < f[30, 25] < 0.4  # exp(-5/4)
    assert np.all(np.diff(f[25:, 25]) <= 0)


def test_drawing_ink_and_alpha_png():
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[5:10, 5:10] = [0, 0, 0, 255]  # opaque black square, rest transparent
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    rgb = load_rgb(buf.getvalue())
    assert rgb[0, 0].tolist() == [255, 255, 255]
    ink = ink_from_drawing(rgb)
    assert ink[7, 7] == 1.0 and ink[0, 0] == 0.0
