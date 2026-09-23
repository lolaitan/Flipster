"""NumPy reference implementation.

This is the ground truth the C++ and CUDA backends are tested against, so every
operation is written to match the native code exactly: replicate (clamp-to-edge)
borders everywhere, float32 math, a hand-written bilinear sampler (``cv2.remap``
uses 5-bit fixed-point weights, which would not match), and the same order of
operations in the 2x2 Lucas-Kanade solve.

Pipeline per frame pair:
    1. Gaussian pyramids of both inputs.
    2. Coarse-to-fine: at each level, warp I1 toward I0 with the current flow,
       accumulate windowed normal equations, solve the regularized 2x2 system,
       repeat ``iterations`` times, then 3x3-median the flow and upsample it.
    3. Forward-backward consistency gives a per-pixel reliability.
    4. In-betweens are made by forward (softmax) splatting both frames to time t
       and blending them by temporal distance x reliability.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .params import FlowParams, SplatParams

_G5 = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


# --------------------------------------------------------------------------- basics


def gauss5(img: np.ndarray) -> np.ndarray:
    return cv2.sepFilter2D(img, cv2.CV_32F, _G5, _G5, borderType=cv2.BORDER_REPLICATE)


def downsample2(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(gauss5(img)[::2, ::2])


def build_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [gauss5(img.astype(np.float32))]
    for _ in range(1, levels):
        pyr.append(downsample2(pyr[-1]))
    return pyr


def gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences, replicate border, correctly scaled (1/2).

    (v1 used cv2.Sobel(ksize=3), which is 8x the derivative; with an unscaled
    temporal term that made every flow vector 8x too small.)
    """
    p = np.pad(img, 1, mode="edge")
    ix = (p[1:-1, 2:] - p[1:-1, :-2]) * np.float32(0.5)
    iy = (p[2:, 1:-1] - p[:-2, 1:-1]) * np.float32(0.5)
    return ix, iy


def box_mean(img: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.boxFilter(img, cv2.CV_32F, (k, k), normalize=True, borderType=cv2.BORDER_REPLICATE)


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample with clamp-to-edge. ``img`` is (H, W) or (H, W, C)."""
    h, w = img.shape[:2]
    x = np.clip(x, 0.0, w - 1).astype(np.float32)
    y = np.clip(y, 0.0, h - 1).astype(np.float32)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def median3x3(img: np.ndarray) -> np.ndarray:
    p = np.pad(img, 1, mode="edge")
    h, w = img.shape
    stack = np.stack([p[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)])
    return np.median(stack, axis=0).astype(np.float32)


def upsample_flow(u: np.ndarray, v: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear x2 upsampling (pixel-center aligned) with vectors scaled by 2."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (xs + 0.5) * 0.5 - 0.5
    cy = (ys + 0.5) * 0.5 - 0.5
    return 2.0 * sample_bilinear(u, cx, cy), 2.0 * sample_bilinear(v, cx, cy)


# --------------------------------------------------------------------------- flow

COARSE_ITER_MULT = 3


def lk_level(i0, i1, u, v, p: FlowParams, iterations: int):
    h, w = i0.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ix, iy = gradients(i0)
    ixx, ixy, iyy = ix * ix, ix * iy, iy * iy
    r = p.window_radius
    step = np.float32(p.max_step)
    for _ in range(iterations):
        wx, wy = xs + u, ys + v
        # Samples that land outside I1 carry no information, so they are dropped
        # from both sides of the normal equations (masked LK).
        m = ((wx >= 0) & (wx <= w - 1) & (wy >= 0) & (wy <= h - 1)).astype(np.float32)
        it = sample_bilinear(i1, wx, wy) - i0
        # Residual linearized about each pixel's *own* current flow:
        #   I1(y + d) ~= I0(y) + It(y) + g(y).(d - u(y))
        # so the window at x solves (G + lam I) d = lam u - sum_y g(y) (It(y) - g(y).u(y)).
        # This is the per-window LK solution; unlike the naive "box(g * It)" update it
        # is a structure-weighted average of neighbour flows, which keeps the coupled
        # dense iteration stable (the naive update amplifies high-frequency noise).
        res = m * (it - (ix * u + iy * v))
        gxx = box_mean(m * ixx, r)
        gxy = box_mean(m * ixy, r)
        gyy = box_mean(m * iyy, r)
        bx = box_mean(ix * res, r)
        by = box_mean(iy * res, r)
        # Levenberg-Marquardt damping toward the current estimate, scaled by the
        # window's gradient energy so every pyramid level converges at the same
        # rate, plus a small floor for flat areas. ``zero_pull`` (mu) is a weak prior
        # toward no motion for directions the window cannot observe.
        lam = np.float32(p.damping) * np.float32(0.5) * (gxx + gyy) + np.float32(p.damping_floor)
        mu = np.float32(p.zero_pull)
        a = gxx + lam + mu
        c = gyy + lam + mu
        det = a * c - gxy * gxy
        rx = lam * u - bx
        ry = lam * v - by
        nu = (c * rx - gxy * ry) / det
        nv = (a * ry - gxy * rx) / det
        u = u + np.clip(nu - u, -step, step)
        v = v + np.clip(nv - v, -step, step)
    if p.median:
        u, v = median3x3(u), median3x3(v)
    return u, v


def pyramidal_lk(i0: np.ndarray, i1: np.ndarray, p: FlowParams | None = None) -> np.ndarray:
    """Dense flow from ``i0`` to ``i1`` (both HxW float in [0, 1]). Returns HxWx2."""
    p = p or FlowParams()
    levels = p.resolve_levels(*i0.shape)
    pyr0 = build_pyramid(i0, levels)
    pyr1 = build_pyramid(i1, levels)
    u = np.zeros_like(pyr0[-1])
    v = np.zeros_like(pyr0[-1])
    for lvl in range(levels - 1, -1, -1):
        if u.shape != pyr0[lvl].shape:
            u, v = upsample_flow(u, v, pyr0[lvl].shape)
        # The coarsest level starts from zero and is tiny, so it gets extra iterations.
        iters = p.iterations * (COARSE_ITER_MULT if lvl == levels - 1 else 1)
        u, v = lk_level(pyr0[lvl], pyr1[lvl], u, v, p, iters)
    return np.dstack([u, v]).astype(np.float32)


# --------------------------------------------------------------------------- synthesis


def warp_backward(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """out(x) = img(x + flow(x))."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return sample_bilinear(img, xs + flow[..., 0], ys + flow[..., 1])


def fb_reliability(f01: np.ndarray, f10: np.ndarray, sp: SplatParams) -> np.ndarray:
    """Soft forward-backward consistency of f01 (per pixel of frame 0)."""
    back = warp_backward(f10, f01)
    err = np.sum((f01 + back) ** 2, axis=-1)
    thr = sp.fb_alpha * (np.sum(f01**2, axis=-1) + np.sum(back**2, axis=-1)) + sp.fb_beta
    return np.exp(-err / thr).astype(np.float32)


def splat(color: np.ndarray, weight: np.ndarray, rel: np.ndarray, flow: np.ndarray, t: float):
    """Bilinear forward splatting of ``color`` along ``t * flow``.

    Returns accumulated (weight*color, weight, weight*reliability) buffers.
    """
    h, w, c = color.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = xs + np.float32(t) * flow[..., 0]
    qy = ys + np.float32(t) * flow[..., 1]
    x0 = np.floor(qx).astype(np.int64)
    y0 = np.floor(qy).astype(np.int64)
    fx = (qx - x0).astype(np.float32)
    fy = (qy - y0).astype(np.float32)
    acc_c = np.zeros((h * w, c), np.float64)
    acc_w = np.zeros(h * w, np.float64)
    acc_r = np.zeros(h * w, np.float64)
    flat_c = color.reshape(-1, c)
    for dy, dx, bw in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        tx, ty = x0 + dx, y0 + dy
        ok = ((tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)).ravel()
        idx = (ty * w + tx).ravel()[ok]
        ww = (weight * bw).ravel()[ok]
        acc_w += np.bincount(idx, ww, minlength=h * w)
        acc_r += np.bincount(idx, ww * rel.ravel()[ok], minlength=h * w)
        for ch in range(c):
            acc_c[:, ch] += np.bincount(idx, ww * flat_c[ok, ch], minlength=h * w)
    return acc_c.reshape(h, w, c), acc_w.reshape(h, w), acc_r.reshape(h, w)


COVERAGE_EPS = 0.05


def compose(c0, c1, s0, s1, t: float, sp: SplatParams) -> np.ndarray:
    """Blend the two splatted frames; fall back to a cross-dissolve on holes."""
    (a0c, a0w, a0r), (a1c, a1w, a1r) = s0, s1
    cov0 = a0w > COVERAGE_EPS
    cov1 = a1w > COVERAGE_EPS
    safe0 = np.where(cov0, a0w, 1.0)
    safe1 = np.where(cov1, a1w, 1.0)
    m = sp.min_reliability
    k0 = (1 - t) * (m + (1 - m) * a0r / safe0) * cov0
    k1 = t * (m + (1 - m) * a1r / safe1) * cov1
    tot = k0 + k1
    has = tot > 0
    out = (k0[..., None] * a0c / safe0[..., None] + k1[..., None] * a1c / safe1[..., None]) / np.where(has, tot, 1.0)[..., None]
    fallback = (1 - t) * c0 + t * c1
    return np.where(has[..., None], out, fallback).astype(np.float32)


# --------------------------------------------------------------------------- engine


class NumpyEngine:
    """Reference engine. Same interface as the native ``flipster._core`` engines."""

    name = "numpy"

    def __init__(self) -> None:
        self.last_timings: dict[str, float] = {}
        self._pair = None

    def flow(self, a: np.ndarray, b: np.ndarray, params: FlowParams | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        f = pyramidal_lk(np.asarray(a, np.float32), np.asarray(b, np.float32), params)
        self.last_timings = {"flow_ms": (time.perf_counter() - t0) * 1e3}
        return f

    def warp(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        img = np.asarray(img, np.float32)
        squeeze = img.ndim == 2
        out = warp_backward(img[..., None] if squeeze else img, flow)
        return out[..., 0] if squeeze else out

    def set_pair(self, c0, c1, imp0, imp1, f01, f10, sp: SplatParams | None = None) -> None:
        t0 = time.perf_counter()
        sp = sp or SplatParams()
        c0 = np.asarray(c0, np.float32)
        c1 = np.asarray(c1, np.float32)
        if c0.ndim == 2:
            c0, c1 = c0[..., None], c1[..., None]
        h, w = c0.shape[:2]
        imp0 = np.zeros((h, w), np.float32) if imp0 is None else np.asarray(imp0, np.float32)
        imp1 = np.zeros((h, w), np.float32) if imp1 is None else np.asarray(imp1, np.float32)
        r0 = fb_reliability(f01, f10, sp)
        r1 = fb_reliability(f10, f01, sp)
        self._pair = dict(
            c0=c0, c1=c1, w0=np.exp(sp.softmax_beta * imp0), w1=np.exp(sp.softmax_beta * imp1),
            r0=r0, r1=r1, f01=np.asarray(f01, np.float32), f10=np.asarray(f10, np.float32), sp=sp,
        )
        self.last_timings = {"set_pair_ms": (time.perf_counter() - t0) * 1e3}

    def reliability(self) -> tuple[np.ndarray, np.ndarray]:
        return self._pair["r0"], self._pair["r1"]

    def synthesize(self, t: float) -> np.ndarray:
        if self._pair is None:
            raise RuntimeError("call set_pair() first")
        t0 = time.perf_counter()
        p = self._pair
        s0 = splat(p["c0"], p["w0"], p["r0"], p["f01"], t)
        s1 = splat(p["c1"], p["w1"], p["r1"], p["f10"], 1.0 - t)
        out = compose(p["c0"], p["c1"], s0, s1, t, p["sp"])
        self.last_timings = {"synth_ms": (time.perf_counter() - t0) * 1e3}
        return out
