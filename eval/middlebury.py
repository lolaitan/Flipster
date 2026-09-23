#!/usr/bin/env python3
"""Accuracy on the Middlebury optical-flow benchmark (Baker et al., IJCV 2011).

Two questions, answered on the public training sequences:

1. Is the flow right?  Average end-point error (AEE) and angular error (AAE)
   against ground-truth flow for frame10 -> frame11.
2. Are the in-betweens right?  PSNR / SSIM of the synthesized midpoint against
   the ground-truth frame10i11 that Middlebury provides for interpolation.

    python eval/middlebury.py --download          # ~30 MB into eval/data
    python eval/middlebury.py --backend cuda --out eval/results.md

Natural video is not what Flipster is tuned for (it expects line art), so this
is a sanity check of the core algorithm against well-known references, not a
leaderboard entry.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import flipster  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "python"))

from flipster import FlowParams, SplatParams, get_engine  # noqa: E402
from flipster.reference import warp_backward  # noqa: E402

BASE = "https://vision.middlebury.edu/flow/data/comp/zip/"
ARCHIVES = ("other-data.zip", "other-gt-flow.zip", "other-gt-interp.zip")
TAG_FLOAT = 202021.25
UNKNOWN = 1e9


# --------------------------------------------------------------------------- io


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        print(f"downloading {name} ...", flush=True)
        with urllib.request.urlopen(BASE + name, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)


def read_flo(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        tag = np.frombuffer(f.read(4), np.float32)[0]
        if tag != TAG_FLOAT:
            raise ValueError(f"{path}: not a .flo file")
        w, h = np.frombuffer(f.read(8), np.int32)
        return np.frombuffer(f.read(), np.float32).reshape(h, w, 2).copy()


def write_flo(path: Path, flow: np.ndarray) -> None:
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        f.write(np.float32(TAG_FLOAT).tobytes())
        f.write(np.int32([w, h]).tobytes())
        f.write(flow.astype(np.float32).tobytes())


def gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- metrics


def flow_errors(est: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    valid = (np.abs(gt) < UNKNOWN).all(-1)
    e, g = est[valid], gt[valid]
    aee = float(np.mean(np.linalg.norm(e - g, axis=-1)))
    num = 1.0 + (e * g).sum(-1)
    den = np.sqrt(1.0 + (e**2).sum(-1)) * np.sqrt(1.0 + (g**2).sum(-1))
    aae = float(np.degrees(np.mean(np.arccos(np.clip(num / den, -1.0, 1.0)))))
    return aee, aae


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM (Wang et al. 2004) on luminance with an 11x11 Gaussian window."""
    if a.ndim == 3:
        a, b = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    c1, c2 = 0.01**2, 0.03**2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a**2
    sbb = blur(b * b) - mu_b**2
    sab = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))
    return float(s.mean())


# --------------------------------------------------------------------------- methods


def reference_flows():
    def farneback(a, b):
        return cv2.calcOpticalFlowFarneback(
            (a * 255).astype(np.uint8), (b * 255).astype(np.uint8), None, 0.5, 5, 15, 3, 5, 1.2, 0
        )

    def dis(a, b):
        d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return d.calc((a * 255).astype(np.uint8), (b * 255).astype(np.uint8), None)

    return {"OpenCV Farneback": farneback, "OpenCV DIS (medium)": dis}


@dataclass
class FlowRow:
    seq: str
    method: str
    aee: float
    aae: float


@dataclass
class InterpRow:
    seq: str
    method: str
    psnr: float
    ssim: float


def evaluate_flow(root: Path, backend: str, params: FlowParams) -> list[FlowRow]:
    rows: list[FlowRow] = []
    eng = get_engine(backend)
    methods = {f"Flipster LK ({eng.name})": lambda a, b: eng.flow(a, b, params), **reference_flows()}
    for gt_path in sorted((root / "other-gt-flow").glob("*/flow10.flo")):
        seq = gt_path.parent.name
        d = root / "other-data" / seq
        if not (d / "frame10.png").exists():
            continue
        a, b, gt = gray(d / "frame10.png"), gray(d / "frame11.png"), read_flo(gt_path)
        for name, fn in methods.items():
            aee, aae = flow_errors(fn(a, b), gt)
            rows.append(FlowRow(seq, name, aee, aae))
    return rows


def evaluate_interp(root: Path, backend: str, params: FlowParams) -> list[InterpRow]:
    rows: list[InterpRow] = []
    eng = get_engine(backend)
    for gt_path in sorted((root / "other-gt-interp").glob("*/frame10i11.png")):
        seq = gt_path.parent.name
        d = root / "other-data" / seq
        if not (d / "frame10.png").exists():
            continue
        i0, i1, gt = rgb(d / "frame10.png"), rgb(d / "frame11.png"), rgb(gt_path)
        g0, g1 = cv2.cvtColor(i0, cv2.COLOR_RGB2GRAY), cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
        f01, f10 = eng.flow(g0, g1, params), eng.flow(g1, g0, params)
        preds = {
            "Cross-dissolve": 0.5 * i0 + 0.5 * i1,
            # v1's approach with its sign and scale bugs fixed: backward-warp both
            # frames halfway using flow sampled at the *target* pixel.
            "Backward warp (v1 method, fixed)": 0.5 * warp_backward(i0, -0.5 * f01)
            + 0.5 * warp_backward(i1, -0.5 * f10),
        }
        eng.set_pair(i0, i1, None, None, f01, f10, SplatParams())
        preds[f"Flipster splat ({eng.name})"] = eng.synthesize(0.5)
        for name, pred in preds.items():
            pred = np.clip(pred, 0, 1).astype(np.float32)
            rows.append(InterpRow(seq, name, psnr(pred, gt), ssim(pred, gt)))
    return rows


# --------------------------------------------------------------------------- report


def _table(rows, key_a: str, key_b: str, fmt: str, label_a: str, label_b: str) -> list[str]:
    seqs = sorted({r.seq for r in rows})
    methods = list(dict.fromkeys(r.method for r in rows))
    out = [f"| method | {' | '.join(seqs)} | **mean {label_a}** | mean {label_b} |", "|---" * (len(seqs) + 3) + "|"]
    for m in methods:
        mine = {r.seq: r for r in rows if r.method == m}
        cells = [format(getattr(mine[s], key_a), fmt) if s in mine else "—" for s in seqs]
        ma = np.mean([getattr(r, key_a) for r in mine.values()])
        mb = np.mean([getattr(r, key_b) for r in mine.values()])
        out.append(f"| {m} | {' | '.join(cells)} | **{format(ma, fmt)}** | {format(mb, fmt)} |")
    return out


def report(flow_rows: list[FlowRow], interp_rows: list[InterpRow]) -> str:
    lines = ["### Flow accuracy (frame10 → frame11)", "", "Average end-point error in pixels (lower is better).", ""]
    lines += _table(flow_rows, "aee", "aae", ".3f", "AEE", "AAE°")
    lines += ["", "### Midpoint interpolation (frame10i11)", "", "PSNR in dB (higher is better).", ""]
    lines += _table(interp_rows, "psnr", "ssim", ".2f", "PSNR", "SSIM")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "eval" / "data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.download:
        download(args.data)
    if not (args.data / "other-data").exists():
        sys.exit(f"no data in {args.data}; run with --download first")
    params = FlowParams(damping=0.05, median=True)
    md = report(evaluate_flow(args.data, args.backend, params), evaluate_interp(args.data, args.backend, params))
    print(md)
    if args.out:
        args.out.write_text(md)


if __name__ == "__main__":
    main()
