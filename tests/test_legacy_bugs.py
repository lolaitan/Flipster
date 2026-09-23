"""Pins down the three v1 bugs and shows the rewrite fixes each one."""

import cv2
import numpy as np

from flipster import FlowParams, legacy
from flipster.legacy import lucas_kanade_v1
from flipster.reference import pyramidal_lk, warp_backward


def blob(cx, h=48, w=72, s=5.0):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - cx) ** 2 + (yy - h // 2) ** 2) / (2 * s * s)).astype(np.float32)


def centroid_x(img):
    w = img.sum(0)
    return float((w * np.arange(img.shape[1])).sum() / w.sum())


def test_v1_flow_is_eight_times_too_small():
    a, b = (blob(34) * 255).astype(np.uint8), (blob(36) * 255).astype(np.uint8)
    flow = lucas_kanade_v1(a, b)
    est = flow[..., 0][a > 60].mean()
    assert 0.15 < est < 0.4  # true motion is 2.0 px -> 2 / 8 = 0.25
    fixed = pyramidal_lk(blob(34), blob(36), FlowParams(levels=1, iterations=10))
    assert abs(fixed[..., 0][a > 60].mean() - 2.0) < 0.15


def test_v1_warp_direction_is_backwards(monkeypatch):
    a, b = blob(34, s=2.5), blob(46, s=2.5)  # moves +12 px, so the midpoint is at 40

    def exact_flow(img1, img2):  # hand v1 the *correct* flow, so only its warp is under test
        flow = np.zeros(img1.shape + (2,))
        flow[..., 0] = centroid_x(img2.astype(np.float64)) - centroid_x(img1.astype(np.float64))
        return flow

    def to_bgr(img):
        return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    monkeypatch.setattr(legacy, "lucas_kanade_v1", exact_flow)
    v1 = legacy.interpolate_v1(to_bgr(a), to_bgr(b))[..., 0].astype(np.float32)
    profile = v1.sum(0)
    assert abs(int(profile.argmax()) - 40) > 8  # two copies, pushed away from the midpoint (to 28 and 52)
    assert profile[40] < 0.1 * profile.max()

    flow = np.zeros(a.shape + (2,), np.float32)
    flow[..., 0] = 12.0
    fixed = warp_backward(a, -0.5 * flow)
    assert abs(centroid_x(fixed) - 40) < 0.3


def test_single_level_needs_small_motion():
    a, b = blob(50, h=128, w=192, s=8), blob(70, h=128, w=192, s=8)  # 20 px
    one = pyramidal_lk(a, b, FlowParams(levels=1))
    pyr = pyramidal_lk(a, b, FlowParams())
    m = a > 0.3
    assert abs(one[..., 0][m].mean() - 20) > 5
    assert abs(pyr[..., 0][m].mean() - 20) < 1.0
