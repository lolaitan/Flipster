#!/usr/bin/env python3
"""Optical-flow throughput benchmark: v1 Python loop vs NumPy vs C++/OpenMP vs CUDA.

Every method gets the same synthetic multi-scale texture translated by a known
amount, so the table also reports end-point error (EPE): a fast backend that
computes the wrong thing would show up immediately.

    python bench/run_bench.py                 # 480p / 720p / 1080p
    python bench/run_bench.py --sizes 4k      # add 3840x2160
    python bench/run_bench.py --reps 10 --out bench/results/my-machine.md

Timing notes
* "wall" includes host<->device copies and Python/pybind overhead;
  "compute" is the backend's own timer (CUDA events for the GPU).
* The v1 loop is timed on a small crop and scaled linearly by pixel count
  (it is O(pixels x window^2) with no pyramid). It is marked "est.".
* OpenCV Farneback / DIS are included as widely known dense-flow reference points,
  not as the same algorithm.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
try:
    import flipster  # noqa: F401
except ImportError:  # running from a checkout without `pip install .`
    sys.path.insert(0, str(ROOT / "python"))

from synth import multiscale_texture  # noqa: E402

from flipster import FlowParams, available_backends, device_info, get_engine  # noqa: E402
from flipster.legacy import lucas_kanade_v1  # noqa: E402

SIZES = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
SHIFT = (12.5, -7.25)


def make_pair(w: int, h: int):
    pad = 64
    big = multiscale_texture(h + 2 * pad, w + 2 * pad, seed=11)
    m = np.float32([[1, 0, SHIFT[0]], [0, 1, SHIFT[1]]])
    moved = cv2.warpAffine(big, m, big.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return big[pad : pad + h, pad : pad + w].copy(), moved[pad : pad + h, pad : pad + w].copy()


def epe(flow: np.ndarray) -> float:
    h, w = flow.shape[:2]
    m = max(16, min(h, w) // 8)
    inner = flow[m:-m, m:-m]
    return float(np.median(np.hypot(inner[..., 0] - SHIFT[0], inner[..., 1] - SHIFT[1])))


def time_it(fn, reps: int):
    fn()  # warm-up: allocations, CUDA context, caches
    walls, computes, out = [], [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out, compute = fn()
        walls.append((time.perf_counter() - t0) * 1e3)
        computes.append(compute if compute is not None else walls[-1])
    return statistics.median(walls), statistics.median(computes), out


def run(sizes: list[str], reps: int, include_v1: bool) -> dict:
    results: dict = {"machine": machine_info(), "shift": SHIFT, "rows": []}
    label = {
        "numpy": "NumPy reference (vectorized)",
        "cpu": f"C++ / OpenMP ({os.cpu_count()} threads)",
        "cuda": "CUDA, shared-memory box filter",
        "cuda-naive": "CUDA, naive box filter",
    }
    backends = [b for b in ("numpy", "cpu", "cuda") if b in available_backends()]
    engines = {label[b]: get_engine(b) for b in backends}
    if "cuda" in backends:
        engines[label["cuda-naive"]] = get_engine("cuda", variant="naive")
    params = FlowParams()

    for name in sizes:
        w, h = SIZES[name]
        a, b = make_pair(w, h)
        row = {"size": name, "width": w, "height": h, "methods": {}}
        print(f"\n== {name} ({w}x{h})", flush=True)

        if include_v1:
            cw, ch = 128, 96
            ca, cb = (a[:ch, :cw] * 255).astype(np.uint8), (b[:ch, :cw] * 255).astype(np.uint8)
            t0 = time.perf_counter()
            lucas_kanade_v1(ca, cb)
            per_px = (time.perf_counter() - t0) * 1e3 / (cw * ch)
            est = per_px * w * h
            row["methods"]["v1 (15-112) Python loop"] = {
                "wall_ms": est,
                "compute_ms": est,
                "epe": None,
                "estimated": True,
            }
            print(f"  v1 (15-112) Python loop          ~{est / 1000:8.1f} s (est.)", flush=True)

        for label, eng in engines.items():
            slow = label.startswith("NumPy")
            n = 1 if slow and w * h > 2e6 else (max(2, reps // 3) if slow else reps)

            def call(e=eng, a=a, b=b):
                f = e.flow(a, b, params)
                t = e.last_timings
                return f, t.get("kernel_ms", t.get("flow_ms"))

            wall, compute, flow = time_it(call, n)
            row["methods"][label] = {"wall_ms": wall, "compute_ms": compute, "epe": epe(flow)}
            print(f"  {label:<32} {wall:9.2f} ms wall  {compute:9.2f} ms compute  EPE {epe(flow):.3f}", flush=True)

        a8, b8 = (a * 255).astype(np.uint8), (b * 255).astype(np.uint8)

        def farneback(a8=a8, b8=b8):
            return cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 5, 15, 3, 5, 1.2, 0), None

        def dis(a8=a8, b8=b8):
            return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a8, b8, None), None

        for label, fn in (("OpenCV Farneback", farneback), ("OpenCV DIS (medium)", dis)):
            wall, _, flow = time_it(fn, max(2, reps // 2))
            row["methods"][label] = {"wall_ms": wall, "compute_ms": wall, "epe": epe(flow), "reference": True}
            print(f"  {label:<32} {wall:9.2f} ms wall                     EPE {epe(flow):.3f}", flush=True)
        results["rows"].append(row)
    return results


def machine_info() -> dict:
    info = device_info()
    cpu = platform.processor() or platform.machine()
    try:
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cpu_threads": os.cpu_count(),
        "gpu": info.get("cuda_device"),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "platform": platform.platform(),
    }


def to_markdown(res: dict) -> str:
    m = res["machine"]
    lines = [
        f"CPU: {m['cpu']} ({m['cpu_threads']} threads) · GPU: {m['gpu'] or 'none'} · {m['platform']}",
        "",
        "Median time for one dense flow field (both frames already in memory). "
        "EPE = median end-point error vs. the known shift, in pixels.",
        "",
    ]
    methods: list[str] = []
    for row in res["rows"]:
        for k in row["methods"]:
            if k not in methods:
                methods.append(k)
    header = "| method | " + " | ".join(r["size"] for r in res["rows"]) + " | EPE |"
    lines += [header, "|---" * (len(res["rows"]) + 2) + "|"]
    for k in methods:
        cells = []
        for r in res["rows"]:
            v = r["methods"].get(k)
            if not v:
                cells.append("—")
                continue
            ms = v["compute_ms"] if k.startswith("CUDA") else v["wall_ms"]
            txt = f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.1f} ms"
            cells.append(("~" + txt + " (est.)") if v.get("estimated") else txt)
        epes = [r["methods"][k]["epe"] for r in res["rows"] if k in r["methods"] and r["methods"][k]["epe"] is not None]
        e = f"{max(epes):.3f}" if epes else "—"
        lines.append(f"| {k} | " + " | ".join(cells) + f" | {e} |")
    if any(k.startswith("CUDA") for k in methods):
        lines += ["", "CUDA rows are kernel time (CUDA events); see the JSON for wall time including PCIe copies."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="480p,720p,1080p", help=f"comma list from {list(SIZES)}")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-v1", action="store_true", help="skip the (estimated) v1 Python loop")
    ap.add_argument("--out", type=Path, default=None, help="write markdown here (and .json next to it)")
    args = ap.parse_args()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    res = run(sizes, args.reps, not args.no_v1)
    md = to_markdown(res)
    print("\n" + md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        args.out.with_suffix(".json").write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
