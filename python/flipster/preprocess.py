"""Turning phone scans of notebook pages (or browser drawings) into clean, aligned ink.

Scans of flipbook pages have problems that break optical flow before it starts:
every page is cropped and lit differently, the paper has blue ruled lines, a red
margin, hole punches, and ink bleeding through from the other side. So before
computing any flow we

1. resize every page to a common size,
2. extract the ink: ``max(R, G, B)`` makes blue rules / red margins / blue stains
   almost white, then dividing by a local paper-brightness estimate removes
   lighting gradients,
3. drop hole punches (thick, round blobs hugging the left/right edge),
4. stabilise the pages: consecutive pages are aligned on the printed paper
   (ruled lines / margin), so scan jitter is removed and the remaining motion is
   the animation itself,
5. build the flow input: ``exp(-d / sigma)`` where ``d`` is the distance to the
   nearest stroke. A 2px pencil line has almost no support for a 15x15 LK window;
   the distance field turns it into a wide, smooth ridge LK can lock onto.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------- loading


def load_rgb(src: bytes | str) -> np.ndarray:
    """Load an image as uint8 RGB, honouring EXIF rotation and compositing alpha on white."""
    im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"))


def working_size(shape: tuple[int, ...], working_height: int) -> tuple[int, int]:
    h, w = shape[:2]
    height = min(working_height, h)
    return max(8, round(w * height / h)), height  # (W, H) for cv2


# --------------------------------------------------------------------------- ink


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ink_from_scan(rgb: np.ndarray, lo: float = 0.18, hi: float = 0.5) -> np.ndarray:
    """Ink strength in [0, 1] for a photographed/scanned page."""
    h = rgb.shape[0]
    v = rgb.max(axis=2).astype(np.float32) / 255.0
    k = _odd(0.012 * h)
    paper = cv2.dilate(v, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    paper = cv2.GaussianBlur(paper, (0, 0), 0.01 * h)
    darkness = np.clip((paper - v) / np.maximum(paper, 1e-3), 0.0, 1.0)
    return _smoothstep(lo, hi, darkness).astype(np.float32)


def ink_from_drawing(rgb: np.ndarray) -> np.ndarray:
    """Ink strength for a clean digital drawing (dark strokes on white)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.clip((1.0 - gray) * 1.25, 0.0, 1.0)


def remove_hole_punches(ink: np.ndarray) -> np.ndarray:
    """Zero out round, filled blobs near the left/right page edge (binder holes)."""
    h, w = ink.shape
    r = max(2, round(0.005 * h))
    mask = (ink > 0.5).astype(np.uint8)
    thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(thick)
    out = ink.copy()
    min_area, max_area = np.pi * (0.006 * h) ** 2, np.pi * (0.03 * h) ** 2
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx = cents[i][0]
        near_edge = cx < 0.12 * w or cx > 0.88 * w
        round_ish = 0.6 < bw / max(bh, 1) < 1.6 and area > 0.5 * bw * bh
        if near_edge and round_ish and min_area <= area <= max_area:
            blob = (labels == i).astype(np.uint8)
            blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1)))
            out[blob > 0] = 0.0
    return out


def soft_field(ink: np.ndarray, sigma: float) -> np.ndarray:
    """exp(-distance_to_ink / sigma): the flow input for line art."""
    strokes = (ink > 0.35).astype(np.uint8)
    if not strokes.any():
        return np.zeros_like(ink, dtype=np.float32)
    dist = cv2.distanceTransform(1 - strokes, cv2.DIST_L2, 5)
    return np.maximum(np.exp(-dist / sigma), ink).astype(np.float32)


# --------------------------------------------------------------------------- registration


def _profile_shift(pa: np.ndarray, pb: np.ndarray, max_shift: int) -> tuple[int, float]:
    """Integer s maximising corr(pa(x + s), pb(x)) over |s| <= max_shift, plus that correlation."""

    def highpass(p: np.ndarray) -> np.ndarray:
        p = p.astype(np.float32)
        p = p - cv2.GaussianBlur(p.reshape(1, -1), (0, 0), 8).ravel()
        return (p - p.mean()) / (p.std() + 1e-9)

    a, b = highpass(pa), highpass(pb)
    n = len(a)
    best_s, best_c = 0, -1.0
    for sh in range(-max_shift, max_shift + 1):
        x, y = (a[sh:], b[: n - sh]) if sh >= 0 else (a[: n + sh], b[-sh:])
        c = float(np.mean(x * y))
        if c > best_c:
            best_s, best_c = sh, c
    return best_s, best_c


def estimate_paper_shift(prev: np.ndarray, cur: np.ndarray, min_corr: float = 0.5) -> tuple[float, float]:
    """Translation that lines ``cur`` up with ``prev``, measured on the *paper*, not the drawing.

    Registering on the ink does not work for flipbooks: pages are hand-traced, so
    the "static" scenery is never identical and the character is supposed to move
    (affine ECC explained the walk as a page stretch; phase correlation on ink
    gave shifts of 40-110 px). The printed paper, however, is identical on every
    page. Its colourful structure (``max - min`` over RGB: blue rules, red margin,
    none of the grey pencil) is projected onto each axis and the 1-D profiles are
    cross-correlated - a classic document-image registration trick. On the
    bundled scans this recovers the 4-14 px margin offsets exactly.

    Returns the sampling offset (dx, dy): aligned(x) = cur(x + offset).
    Axes whose best correlation is below ``min_corr`` (e.g. blank paper) get 0.
    """
    h, w = prev.shape[:2]

    def colourful(rgb: np.ndarray) -> np.ndarray:
        f = rgb.astype(np.float32) / 255.0
        return f.max(axis=2) - f.min(axis=2)

    a, b = colourful(prev), colourful(cur)
    ys, xs = slice(int(0.1 * h), int(0.9 * h)), slice(int(0.15 * w), int(0.85 * w))
    sx, cx = _profile_shift(a[ys].mean(0), b[ys].mean(0), max(4, int(0.04 * w)))
    sy, cy = _profile_shift(a[:, xs].mean(1), b[:, xs].mean(1), max(4, int(0.012 * h)))
    return (-float(sx) if cx >= min_corr else 0.0, -float(sy) if cy >= min_corr else 0.0)


# --------------------------------------------------------------------------- frames


@dataclass
class PreparedFrame:
    rgb: np.ndarray  # float32 HxWx3 in [0, 1], aligned
    ink: np.ndarray  # float32 HxW in [0, 1], aligned
    field: np.ndarray  # float32 HxW, flow input
    transform: np.ndarray  # 2x3 affine applied to the original (resized) page


def prepare_frames(
    images: list[np.ndarray],
    source: str = "scan",
    working_height: int = 1200,
    register: bool = True,
    progress=None,
) -> list[PreparedFrame]:
    """Resize, clean and (optionally) stabilise a sequence of RGB uint8 frames."""
    if not images:
        return []
    size = working_size(images[0].shape, working_height)
    w, h = size
    sigma = max(2.0, 0.006 * h)
    resized = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in images]

    if source == "scan":
        inks = [remove_hole_punches(ink_from_scan(im)) for im in resized]
    elif source == "drawing":
        inks = [ink_from_drawing(im) for im in resized]
    else:  # natural photos/video frames: no ink concept, flow runs on luminance
        inks = [np.zeros((h, w), np.float32) for _ in resized]

    transforms = [np.eye(2, 3, dtype=np.float32) for _ in resized]
    if register and source == "scan" and len(resized) > 1:
        for i in range(1, len(resized)):
            dx, dy = estimate_paper_shift(resized[i - 1], resized[i])
            # Sampling map for page i: aligned_i(x) = raw_i(x + offset_i), chained so
            # every page lines up with its (already aligned) predecessor.
            transforms[i] = transforms[i - 1].copy()
            transforms[i][:, 2] += (dx, dy)
            if progress:
                progress(i / (len(resized) - 1))

    frames = []
    for im, ink, tf in zip(resized, inks, transforms):
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        rgb = cv2.warpAffine(im, tf, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        ink_a = cv2.warpAffine(ink, tf, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgbf = rgb.astype(np.float32) / 255.0
        if source == "photo":
            field = cv2.cvtColor(rgbf, cv2.COLOR_RGB2GRAY)
        else:
            field = soft_field(ink_a, sigma)
        frames.append(PreparedFrame(rgbf, ink_a, field, tf))
    return frames


# --------------------------------------------------------------------------- rendering


PAPER = np.array([0.985, 0.978, 0.955], np.float32)
INK = np.array([0.12, 0.12, 0.16], np.float32)


def render_clean(ink: np.ndarray) -> np.ndarray:
    """Ink on clean paper, float32 HxWx3 in [0, 1]."""
    a = np.clip(ink, 0, 1)[..., None]
    return PAPER * (1 - a) + INK * a


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
