"""End-to-end flipbook in-betweening: pages in, animation frames out."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .backends import get_engine
from .flowviz import flow_to_rgb
from .params import RenderOptions
from .preprocess import PreparedFrame, prepare_frames, render_clean, to_uint8

ProgressFn = Callable[[str, float, str], None]


@dataclass
class OutputFrame:
    image: np.ndarray  # uint8 HxWx3
    key: bool  # True for an original page, False for a generated in-between
    source: int  # index of the page this frame starts from
    t: float  # 0 for key frames, (0, 1) for in-betweens


@dataclass
class PairStats:
    pair: int
    flow_ms: float
    synth_ms: float
    mean_motion_px: float


@dataclass
class RenderResult:
    frames: list[OutputFrame] = field(default_factory=list)
    flow_images: list[np.ndarray] = field(default_factory=list)
    pairs: list[PairStats] = field(default_factory=list)
    backend: str = ""
    width: int = 0
    height: int = 0
    preprocess_ms: float = 0.0
    total_ms: float = 0.0

    def summary(self) -> dict:
        return {
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "frames": len(self.frames) if self.frames else None,
            "preprocess_ms": round(self.preprocess_ms, 1),
            "flow_ms": round(sum(p.flow_ms for p in self.pairs), 1),
            "synth_ms": round(sum(p.synth_ms for p in self.pairs), 1),
            "total_ms": round(self.total_ms, 1),
            "pairs": [p.__dict__ for p in self.pairs],
        }


def _styled(frame: PreparedFrame, style: str) -> np.ndarray:
    return frame.ink[..., None] if style == "clean" else frame.rgb


def _to_rgb8(img: np.ndarray, style: str) -> np.ndarray:
    if style == "clean":
        return to_uint8(render_clean(img[..., 0]))
    return to_uint8(img)


def match_ink_mass(ink: np.ndarray, target: float, iters: int = 24) -> np.ndarray:
    """Thin an in-between so its total ink equals ``target``.

    Softmax splatting lets a stroke win over the paper it lands on, but with
    bilinear splats that also lets it win the half-covered pixels beside it, so
    in-betweens come out ~1 px bolder than the pages and the animation flickers.
    Total ink should interpolate linearly between two pages, so we raise a black
    point until it does: that trims the faint spill and keeps stroke cores.
    """
    if target <= 0 or ink.sum() <= target:
        return ink
    lo, hi = 0.0, 0.95
    for _ in range(iters):
        s = 0.5 * (lo + hi)
        if np.clip((ink - s) / (1 - s), 0, 1).sum() > target:
            lo = s
        else:
            hi = s
    s = 0.5 * (lo + hi)
    return np.clip((ink - s) / (1 - s), 0, 1).astype(np.float32)


def render(
    images: list[np.ndarray],
    opts: RenderOptions | None = None,
    on_progress: ProgressFn | None = None,
    on_frame: Callable[[int, OutputFrame], None] | None = None,
    on_flow: Callable[[int, np.ndarray], None] | None = None,
) -> RenderResult:
    """Generate an in-betweened animation from ordered RGB uint8 pages.

    If ``on_frame`` / ``on_flow`` are given, frames are streamed to them instead of
    being kept in memory (the server writes them straight to disk).
    """
    opts = opts or RenderOptions()
    progress = on_progress or (lambda *_: None)
    t_start = time.perf_counter()
    result = RenderResult()
    if len(images) == 0:
        return result

    progress("preprocess", 0.0, f"Cleaning {len(images)} pages")
    t0 = time.perf_counter()
    frames = prepare_frames(
        images,
        source=opts.source,
        working_height=opts.working_height,
        register=opts.register,
        progress=lambda f: progress("preprocess", f, "Stabilising pages"),
    )
    result.preprocess_ms = (time.perf_counter() - t0) * 1e3
    result.height, result.width = frames[0].ink.shape

    engine = get_engine(opts.backend) if opts.method == "flow" else None
    result.backend = engine.name if engine else "none"
    n_between = opts.inbetweens if opts.method != "none" else 0
    out_index = 0

    def emit(img: np.ndarray, key: bool, src: int, t: float) -> None:
        nonlocal out_index
        f = OutputFrame(img, key, src, t)
        if on_frame:
            on_frame(out_index, f)
        else:
            result.frames.append(f)
        out_index += 1

    n_pairs = len(frames) - 1
    for i in range(len(frames)):
        cur = frames[i]
        emit(_to_rgb8(_styled(cur, opts.style), opts.style), True, i, 0.0)
        if i == n_pairs or n_between == 0:
            continue
        nxt = frames[i + 1]
        c0, c1 = _styled(cur, opts.style), _styled(nxt, opts.style)
        ts = [(k + 1) / (n_between + 1) for k in range(n_between)]
        progress("interpolate", i / max(n_pairs, 1), f"Pair {i + 1}/{n_pairs}")

        if opts.method == "linear":
            s0 = time.perf_counter()
            for t in ts:
                emit(_to_rgb8((1 - t) * c0 + t * c1, opts.style), False, i, t)
            result.pairs.append(PairStats(i, 0.0, (time.perf_counter() - s0) * 1e3, 0.0))
            continue

        f0 = time.perf_counter()
        f01 = engine.flow(cur.field, nxt.field, opts.flow)
        f10 = engine.flow(nxt.field, cur.field, opts.flow)
        flow_ms = (time.perf_counter() - f0) * 1e3

        s0 = time.perf_counter()
        imp0 = cur.ink if opts.source != "photo" else None
        imp1 = nxt.ink if opts.source != "photo" else None
        engine.set_pair(c0, c1, imp0, imp1, f01, f10, opts.splat)
        mass0, mass1 = float(cur.ink.sum()), float(nxt.ink.sum())
        for t in ts:
            out = engine.synthesize(t)
            if opts.style == "clean":
                out = match_ink_mass(out[..., 0], (1 - t) * mass0 + t * mass1)[..., None]
            emit(_to_rgb8(out, opts.style), False, i, t)
        synth_ms = (time.perf_counter() - s0) * 1e3

        ink_mask = cur.ink > 0.35
        motion = float(np.linalg.norm(f01[ink_mask], axis=-1).mean()) if ink_mask.any() else 0.0
        result.pairs.append(PairStats(i, flow_ms, synth_ms, motion))
        # Colour the flow only where there is ink (slightly dilated) so the
        # visualisation reads as "which strokes move where".
        near_ink = cv2.dilate((cur.ink > 0.35).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        viz = flow_to_rgb(f01 * near_ink[..., None]) if opts.source != "photo" else flow_to_rgb(f01)
        if on_flow:
            on_flow(i, viz)
        else:
            result.flow_images.append(viz)

    result.total_ms = (time.perf_counter() - t_start) * 1e3
    progress("done", 1.0, "Done")
    return result
