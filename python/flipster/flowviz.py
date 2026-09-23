"""Optical-flow visualisation using the Middlebury colour wheel (Baker et al. 2011)."""

from __future__ import annotations

import numpy as np


def _color_wheel() -> np.ndarray:
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    cols = []
    cols += [[255, 255 * i / ry, 0] for i in range(ry)]
    cols += [[255 - 255 * i / yg, 255, 0] for i in range(yg)]
    cols += [[0, 255, 255 * i / gc] for i in range(gc)]
    cols += [[0, 255 - 255 * i / cb, 255] for i in range(cb)]
    cols += [[255 * i / bm, 0, 255] for i in range(bm)]
    cols += [[255, 0, 255 - 255 * i / mr] for i in range(mr)]
    return np.array(cols, np.float32) / 255.0


_WHEEL = _color_wheel()


def flow_to_rgb(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """HxWx2 flow -> HxWx3 uint8. Hue = direction, saturation = magnitude."""
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99.5)) if mag.size else 1.0
    max_mag = max(max_mag, 1e-6)
    u, v, mag = u / max_mag, v / max_mag, np.minimum(mag / max_mag, 1.0)
    n = len(_WHEEL)
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1) / 2 * (n - 1)
    k0 = np.floor(fk).astype(int)
    k1 = (k0 + 1) % n
    f = (fk - k0)[..., None]
    col = (1 - f) * _WHEEL[k0] + f * _WHEEL[k1]
    col = 1 - mag[..., None] * (1 - col)
    return (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)
