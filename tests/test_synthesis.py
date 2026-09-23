import numpy as np
import pytest

from flipster import FlowParams, SplatParams, available_backends, get_engine
from synth import stick_figure

BACKENDS = list(available_backends())


def centroid_x(img):
    w = img.sum(0)
    return float((w * np.arange(img.shape[1])).sum() / w.sum())


def figure_pair(shift=24):
    h, w = 120, 200
    a = stick_figure(h, w, 70, 60)
    b = stick_figure(h, w, 70 + shift, 60)
    return a, b


def soft(ink, sigma=4.0):
    import cv2

    d = cv2.distanceTransform((ink < 0.35).astype(np.uint8), cv2.DIST_L2, 5)
    return np.maximum(np.exp(-d / sigma), ink).astype(np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inbetween_is_sharp_and_in_the_middle(backend):
    a, b = figure_pair(24)
    eng = get_engine(backend)
    f01 = eng.flow(soft(a), soft(b))
    f10 = eng.flow(soft(b), soft(a))
    eng.set_pair(a, b, a, b, f01, f10, SplatParams())
    mid = eng.synthesize(0.5)[..., 0]
    assert abs(centroid_x(mid) - (70 + 12)) < 1.0
    # A cross-dissolve leaves two half-strength ghosts; the splatted in-between keeps
    # the strokes at full strength.
    solid = (a > 0.85).sum()
    assert (mid > 0.85).sum() > 0.6 * solid
    assert ((0.5 * a + 0.5 * b) > 0.85).sum() < 0.2 * solid


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("t", [0.25, 0.75])
def test_inbetween_position_follows_t(backend, t):
    a, b = figure_pair(24)
    eng = get_engine(backend)
    eng.set_pair(a, b, a, b, eng.flow(soft(a), soft(b)), eng.flow(soft(b), soft(a)))
    out = eng.synthesize(t)[..., 0]
    assert abs(centroid_x(out) - (70 + 24 * t)) < 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_endpoints_reproduce_keyframes(backend):
    a, b = figure_pair(16)
    eng = get_engine(backend)
    eng.set_pair(a, b, a, b, eng.flow(soft(a), soft(b)), eng.flow(soft(b), soft(a)))
    assert np.abs(eng.synthesize(1e-4)[..., 0] - a).mean() < 0.01
    assert np.abs(eng.synthesize(1 - 1e-4)[..., 0] - b).mean() < 0.01


@pytest.mark.parametrize("backend", BACKENDS)
def test_multichannel_and_uniform_importance(backend):
    a, b = figure_pair(10)
    rgb0 = np.dstack([a, a * 0.5, 1 - a]).astype(np.float32)
    rgb1 = np.dstack([b, b * 0.5, 1 - b]).astype(np.float32)
    eng = get_engine(backend)
    eng.set_pair(rgb0, rgb1, None, None, eng.flow(soft(a), soft(b)), eng.flow(soft(b), soft(a)))
    out = eng.synthesize(0.5)
    assert out.shape == rgb0.shape
    assert np.isfinite(out).all()
    assert out.min() >= -1e-4 and out.max() <= 1 + 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_forward_backward_check_flags_disappearing_content(backend):
    """A mark that exists only in frame 0 has no consistent match."""
    h, w = 100, 140
    a = stick_figure(h, w, 50, 50)
    b = stick_figure(h, w, 58, 50)
    a[70:90, 100:120] = 1.0  # square that vanishes
    eng = get_engine(backend)
    f01, f10 = eng.flow(soft(a), soft(b)), eng.flow(soft(b), soft(a))
    eng.set_pair(a, b, a, b, f01, f10)
    r0, _ = eng.reliability()
    assert r0[75:85, 105:115].mean() < r0[40:60, 45:62].mean()


@pytest.mark.parametrize("backend", BACKENDS)
def test_backward_warp(backend):
    img = np.zeros((20, 30, 1), np.float32)
    img[10, 15] = 1.0
    flow = np.zeros((20, 30, 2), np.float32)
    flow[..., 0] = 2.0  # out(x) = img(x + 2)
    out = get_engine(backend).warp(img, flow)
    assert out[10, 13, 0] == pytest.approx(1.0)
