"""Synthetic test images with known motion."""

import cv2
import numpy as np


def multiscale_texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Noise summed over octaves so every pyramid level has structure. Range [0, 1]."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for sigma in (1.0, 2.0, 4.0, 8.0, 16.0):
        img += cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigma) * sigma
    img -= img.min()
    return img / img.max()


def translated_pair(dx: float, dy: float, h: int = 160, w: int = 200, seed: int = 0):
    """(i0, i1) where the content of i1 is i0's content moved by (+dx, +dy)."""
    pad = 64
    big = multiscale_texture(h + 2 * pad, w + 2 * pad, seed)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(big, m, big.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return big[pad : pad + h, pad : pad + w].copy(), moved[pad : pad + h, pad : pad + w].copy()


def stick_figure(h: int, w: int, cx: float, cy: float, scale: float = 1.0, thickness: int = 2) -> np.ndarray:
    """Ink map (1 = ink) of a little stick figure centred at (cx, cy)."""
    img = np.zeros((h, w), np.uint8)
    s = scale
    p = lambda x, y: (int(round(cx + x * s)), int(round(cy + y * s)))  # noqa: E731
    cv2.circle(img, p(0, -30), int(8 * s), 255, thickness, cv2.LINE_AA)
    cv2.line(img, p(0, -22), p(0, 5), 255, thickness, cv2.LINE_AA)
    cv2.line(img, p(0, -12), p(-12, -2), 255, thickness, cv2.LINE_AA)
    cv2.line(img, p(0, -12), p(12, -2), 255, thickness, cv2.LINE_AA)
    cv2.line(img, p(0, 5), p(-9, 25), 255, thickness, cv2.LINE_AA)
    cv2.line(img, p(0, 5), p(9, 25), 255, thickness, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def ruled_page(h: int, w: int, margin_x: int = 40, spacing: int = 18, offset=(0, 0)) -> np.ndarray:
    """RGB uint8 notebook paper: blue rules, red margin, slight lighting gradient."""
    ox, oy = offset
    yy, xx = np.mgrid[0:h, 0:w]
    base = 245 - 15 * (xx / w)  # uneven lighting
    img = np.dstack([base, base, base]).astype(np.float32)
    rows = ((yy - oy) % spacing) < 1
    img[rows] = [150, 180, 235]
    cols = np.abs(xx - (margin_x + ox)) < 1
    img[cols] = [230, 120, 130]
    return np.clip(img, 0, 255).astype(np.uint8)
