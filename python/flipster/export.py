"""Animated GIF / MP4 export."""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
from PIL import Image


def _fit(img: np.ndarray, max_side: int | None) -> Image.Image:
    im = Image.fromarray(img)
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return im


def to_gif(frames: Iterable[np.ndarray], fps: float = 12, max_side: int | None = 720, loop: bool = True) -> bytes:
    ims = [_fit(f, max_side).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    if not ims:
        raise ValueError("no frames to export")
    buf = io.BytesIO()
    ims[0].save(
        buf, format="GIF", save_all=True, append_images=ims[1:],
        duration=max(20, round(1000 / fps)), loop=0 if loop else 1, disposal=2, optimize=True,
    )
    return buf.getvalue()


def to_mp4(frames: Iterable[np.ndarray], fps: float = 12, max_side: int | None = 1080) -> bytes:
    """H.264 MP4 (needs the ``imageio-ffmpeg`` extra)."""
    import os
    import tempfile

    import imageio.v2 as imageio

    arrs = []
    for f in frames:
        im = _fit(f, max_side)
        w, h = im.size
        arrs.append(np.asarray(im.resize((w - w % 2, h - h % 2))))  # yuv420p needs even dims
    if not arrs:
        raise ValueError("no frames to export")
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2) as w:
            for a in arrs:
                w.append_data(a)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)
