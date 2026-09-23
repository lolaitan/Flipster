"""The v1 (15-112 term project) interpolation, kept verbatim for comparison.

Do not use this for anything but benchmarks and regression tests. It has three
bugs that the rewrite fixes, each pinned by a test in ``tests/test_legacy_bugs.py``:

1. ``cv2.Sobel(ksize=3)`` returns 8x the image derivative while the temporal
   difference is unscaled, so every flow vector comes out 8x too small.
2. The warp samples ``frame(x + t * flow)``, which pushes both frames *away* from
   the midpoint instead of toward it.
3. Single-level, single-iteration LK only linearises correctly for ~1 px of motion,
   which is why v1 had to shrink pages to 1/10 size before running it.

It is also a per-pixel Python loop calling ``np.linalg.cond`` and ``pinv``: about
7 minutes per direction on a full 3500x2760 scan.
"""

from __future__ import annotations

import cv2
import numpy as np


def lucas_kanade_v1(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> np.ndarray:
    img1 = cv2.GaussianBlur(img1, (5, 5), 1)
    img2 = cv2.GaussianBlur(img2, (5, 5), 1)
    ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    it = img2.astype(np.float64) - img1.astype(np.float64)
    flow = np.zeros((img1.shape[0], img1.shape[1], 2))
    half = window_size // 2
    for y in range(half, img1.shape[0] - half):
        for x in range(half, img1.shape[1] - half):
            ix_w = ix[y - half : y + half + 1, x - half : x + half + 1].flatten()
            iy_w = iy[y - half : y + half + 1, x - half : x + half + 1].flatten()
            it_w = it[y - half : y + half + 1, x - half : x + half + 1].flatten()
            a = np.stack((ix_w, iy_w), axis=1)
            b = -it_w
            if np.linalg.cond(a.T @ a) < 1e3:
                flow[y, x] = np.linalg.pinv(a.T @ a) @ a.T @ b
    return flow


def interpolate_v1(frame1: np.ndarray, frame2: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """v1 midpoint synthesis (BGR/RGB uint8 in, same out)."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow12 = lucas_kanade_v1(gray1, gray2)
    flow21 = lucas_kanade_v1(gray2, gray1)
    h, w = gray1.shape
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    map_x1 = (gx + ratio * flow12[..., 0]).astype(np.float32)
    map_y1 = (gy + ratio * flow12[..., 1]).astype(np.float32)
    map_x2 = (gx + (1 - ratio) * flow21[..., 0]).astype(np.float32)
    map_y2 = (gy + (1 - ratio) * flow21[..., 1]).astype(np.float32)
    w1 = cv2.remap(frame1, map_x1, map_y1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    w2 = cv2.remap(frame2, map_x2, map_y2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(w1, 1 - ratio, w2, ratio, 0)
