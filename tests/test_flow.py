import numpy as np
import pytest

from flipster import FlowParams, available_backends, get_engine
from synth import translated_pair

BACKENDS = list(available_backends())


def interior_epe(flow, dx, dy, margin):
    inner = flow[margin:-margin, margin:-margin]
    return np.sqrt(((inner - np.array([dx, dy], np.float32)) ** 2).sum(-1))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dx,dy", [(0.6, -0.3), (7.5, 4.0), (30.0, -18.0)])
def test_translation_is_recovered(backend, dx, dy):
    """Pyramidal LK must recover sub-pixel and large (30 px) motion."""
    i0, i1 = translated_pair(dx, dy, h=240, w=320)
    flow = get_engine(backend).flow(i0, i1, FlowParams())
    epe = interior_epe(flow, dx, dy, margin=40)
    assert np.median(epe) < 0.1, np.median(epe)
    assert np.mean(epe) < 0.35, np.mean(epe)


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_level_cannot_track_large_motion(backend):
    """Sanity check that the pyramid is what makes large motion work."""
    i0, i1 = translated_pair(30.0, -18.0, h=240, w=320)
    flow = get_engine(backend).flow(i0, i1, FlowParams(levels=1))
    assert np.median(interior_epe(flow, 30.0, -18.0, 40)) > 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_identical_frames_give_zero_flow(backend):
    i0, _ = translated_pair(0, 0)
    flow = get_engine(backend).flow(i0, i0)
    assert np.abs(flow).max() < 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_textureless_input_is_stable(backend):
    flat = np.full((64, 80), 0.5, np.float32)
    flow = get_engine(backend).flow(flat, flat)
    assert np.isfinite(flow).all()
    assert np.abs(flow).max() == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_odd_sizes_and_shapes(backend):
    i0, i1 = translated_pair(3.0, 1.0, h=77, w=131)
    flow = get_engine(backend).flow(i0, i1)
    assert flow.shape == (77, 131, 2)
    assert flow.dtype == np.float32


def test_levels_resolution():
    p = FlowParams()
    assert p.resolve_levels(1200, 900) == 6
    assert p.resolve_levels(40, 40) == 2
    assert FlowParams(levels=3).resolve_levels(1200, 900) == 3
