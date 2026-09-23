"""The Middlebury harness, exercised on a tiny synthetic dataset in the same layout."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import middlebury as mb  # noqa: E402
from synth import multiscale_texture  # noqa: E402


def make_dataset(root: Path, shift=(3.0, -2.0)):
    for seq in ("Alpha", "Beta"):
        big = multiscale_texture(200, 260, seed=hash(seq) % 100)

        def img(d, big=big):
            m = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
            return cv2.warpAffine(big, m, (260, 200), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)[
                20:-20, 20:-20
            ]

        frames = {"frame10.png": img((0, 0)), "frame11.png": img(shift)}
        (root / "other-data" / seq).mkdir(parents=True)
        (root / "other-gt-flow" / seq).mkdir(parents=True)
        (root / "other-gt-interp" / seq).mkdir(parents=True)
        for name, f in frames.items():
            cv2.imwrite(str(root / "other-data" / seq / name), (np.dstack([f] * 3) * 255).astype(np.uint8))
        mid = img((shift[0] / 2, shift[1] / 2))
        cv2.imwrite(
            str(root / "other-gt-interp" / seq / "frame10i11.png"), (np.dstack([mid] * 3) * 255).astype(np.uint8)
        )
        gt = np.zeros(mid.shape + (2,), np.float32)
        gt[...] = shift
        gt[:3] = mb.UNKNOWN * 2  # some invalid pixels
        mb.write_flo(root / "other-gt-flow" / seq / "flow10.flo", gt)


def test_flo_roundtrip(tmp_path):
    f = np.random.default_rng(0).normal(size=(7, 9, 2)).astype(np.float32)
    mb.write_flo(tmp_path / "x.flo", f)
    np.testing.assert_array_equal(mb.read_flo(tmp_path / "x.flo"), f)


def test_metrics():
    gt = np.zeros((4, 4, 2), np.float32)
    gt[..., 0] = 1
    assert mb.flow_errors(gt, gt) == (0.0, 0.0)
    aee, _ = mb.flow_errors(gt + [0, 1], gt)
    assert abs(aee - 1.0) < 1e-6
    a = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    assert mb.psnr(a, a) == 99.0 and abs(mb.ssim(a, a) - 1.0) < 1e-6


def test_harness_on_synthetic_data(tmp_path):
    make_dataset(tmp_path)
    flow = mb.evaluate_flow(tmp_path, "numpy", mb.FlowParams())
    ours = [r for r in flow if r.method.startswith("Flipster")]
    assert len(ours) == 2 and all(r.aee < 0.2 for r in ours)
    interp = mb.evaluate_interp(tmp_path, "numpy", mb.FlowParams())
    by = {}
    for r in interp:
        by.setdefault(r.method.split(" (")[0], []).append(r.psnr)
    assert np.mean(by["Flipster splat"]) > np.mean(by["Cross-dissolve"]) + 3
    md = mb.report(flow, interp)
    assert "Flow accuracy" in md and "Alpha" in md
