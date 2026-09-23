#!/usr/bin/env python3
"""Regenerate the README figures from the bundled sample flipbook.

python docs/make_figures.py [--backend cuda]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flipster import RenderOptions, render
from flipster.preprocess import load_rgb

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(img: np.ndarray, text: str, size: int = 26) -> np.ndarray:
    pad = np.full((int(size * 1.7), img.shape[1], 3), 255, np.uint8)
    im = Image.fromarray(np.vstack([pad, img]))
    ImageDraw.Draw(im).text((12, size // 3), text, fill=(30, 31, 36), font=font(size))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    pages = [load_rgb(f) for f in sorted(glob.glob(str(ROOT / "samples" / "scans" / "*.jpg")))]
    flow = render(pages, RenderOptions(method="flow", inbetweens=3, backend=args.backend, working_height=1000))
    lin = render(pages, RenderOptions(method="linear", inbetweens=3, working_height=1000))
    print("rendered", len(flow.frames), "frames in", round(flow.total_ms / 1000, 1), "s on", flow.backend)

    h = flow.height
    y0, y1 = int(0.26 * h), int(0.60 * h)

    def crop(im):
        return im[y0:y1]

    def strip(res, pair):
        fr = [f for f in res.frames if f.source == pair] + [f for f in res.frames if f.source == pair + 1 and f.key]
        return np.hstack([crop(f.image)[:, 60:-20] for f in fr])

    for pair in (4, 13):
        a = label(strip(lin, pair), "Cross-fade (what naive interpolation gives you)")
        b = label(strip(flow, pair), "Flipster: pyramidal LK flow + occlusion-aware splatting")
        out = np.vstack([a, np.full((10, a.shape[1], 3), 235, np.uint8), b])
        s = 1400 / out.shape[1]
        Image.fromarray(out).resize((1400, round(out.shape[0] * s)), Image.LANCZOS).save(
            DOCS / f"inbetween_pair{pair + 1}.png", optimize=True
        )

    keys = [f for f in flow.frames if f.key]
    w = 420

    def small(im):
        c = crop(im)
        return np.asarray(Image.fromarray(c).resize((w, round(c.shape[0] * w / c.shape[1])), Image.LANCZOS))

    frames = []
    for f in flow.frames:
        page = next(k for k in keys if k.source == f.source).image
        left, right = label(small(page), "20 pages"), label(small(f.image), "+ 3 in-betweens per page")
        gap = np.full((left.shape[0], 8, 3), 235, np.uint8)
        frames.append(Image.fromarray(np.hstack([left, gap, right])).convert("P", palette=Image.ADAPTIVE, colors=32))
    frames[0].save(DOCS / "demo.gif", save_all=True, append_images=frames[1:], duration=83, loop=0, optimize=True)

    viz = flow.flow_images[13]
    page = next(k for k in keys if k.source == 13).image
    blend = (page.astype(np.float32) * viz.astype(np.float32) / 255).astype(np.uint8)
    c = crop(blend)[:, 60:-20]
    Image.fromarray(c).resize((900, round(c.shape[0] * 900 / c.shape[1])), Image.LANCZOS).save(
        DOCS / "flow_pair14.png", optimize=True
    )


if __name__ == "__main__":
    main()
