"""Native engines must match the NumPy reference (the executable spec)."""

import numpy as np
import pytest
from synth import stick_figure, translated_pair

from flipster import FlowParams, SplatParams, available_backends, get_engine

NATIVE = [b for b in available_backends() if b != "numpy"]
pytestmark = pytest.mark.skipif(not NATIVE, reason="native extension not built")


@pytest.mark.parametrize("backend", NATIVE)
def test_flow_matches_reference(backend):
    i0, i1 = translated_pair(9.3, -4.1, h=150, w=210, seed=4)
    ref = get_engine("numpy").flow(i0, i1, FlowParams())
    nat = get_engine(backend).flow(i0, i1, FlowParams())
    diff = np.abs(ref - nat)
    assert np.median(diff) < 1e-4
    assert np.mean(diff) < 1e-3


@pytest.mark.parametrize("backend", NATIVE)
def test_synthesis_matches_reference(backend):
    a = stick_figure(110, 170, 60, 55)
    b = stick_figure(110, 170, 85, 58)
    ref, nat = get_engine("numpy"), get_engine(backend)
    f01 = ref.flow(a, b)
    f10 = ref.flow(b, a)
    rgb0 = np.dstack([a, 1 - a, a * 0.3]).astype(np.float32)
    rgb1 = np.dstack([b, 1 - b, b * 0.3]).astype(np.float32)
    for e in (ref, nat):
        e.set_pair(rgb0, rgb1, a, b, f01, f10, SplatParams())
    np.testing.assert_allclose(ref.reliability()[0], nat.reliability()[0], atol=1e-5)
    for t in (0.2, 0.5, 0.9):
        np.testing.assert_allclose(ref.synthesize(t), nat.synthesize(t), atol=1e-4)


@pytest.mark.parametrize("backend", NATIVE)
def test_warp_matches_reference(backend):
    i0, _ = translated_pair(0, 0, h=60, w=90)
    flow = np.dstack([np.sin(np.arange(60 * 90).reshape(60, 90) * 0.01) * 4, np.full((60, 90), -1.5)]).astype(
        np.float32
    )
    np.testing.assert_allclose(get_engine("numpy").warp(i0, flow), get_engine(backend).warp(i0, flow), atol=1e-6)


@pytest.mark.parametrize("backend", NATIVE)
def test_native_reports_timings(backend):
    i0, i1 = translated_pair(2, 1, h=64, w=64)
    e = get_engine(backend)
    e.flow(i0, i1)
    assert e.last_timings["flow_ms"] > 0
