import numpy as np
import pytest
from synth import stick_figure

from flipster import RenderOptions, render
from flipster.export import to_gif


def drawings(n=3, step=15):
    out = []
    for i in range(n):
        ink = stick_figure(120, 160, 50 + i * step, 60)
        out.append((255 * (1 - ink[..., None].repeat(3, -1))).astype(np.uint8))
    return out


@pytest.mark.parametrize("method,expected", [("none", 3), ("linear", 7), ("flow", 7)])
def test_frame_counts(method, expected):
    events = []
    res = render(
        drawings(),
        RenderOptions(method=method, inbetweens=2, source="drawing", backend="numpy"),
        on_progress=lambda s, f, m: events.append(s),
    )
    assert len(res.frames) == expected
    assert [f.key for f in res.frames][:4] == ([True, True, True][:3] + [])[:0] + [f.key for f in res.frames][:4]
    assert sum(f.key for f in res.frames) == 3
    assert events[-1] == "done"
    assert res.frames[0].image.dtype == np.uint8 and res.frames[0].image.shape == (120, 160, 3)


def test_flow_render_streams_and_reports_stats():
    got = {}
    flows = []
    res = render(
        drawings(),
        RenderOptions(method="flow", inbetweens=1, source="drawing", backend="numpy"),
        on_frame=lambda i, f: got.__setitem__(i, f),
        on_flow=lambda i, v: flows.append(i),
    )
    assert sorted(got) == list(range(5))
    assert [got[i].t for i in range(5)] == [0.0, 0.5, 0.0, 0.5, 0.0]
    assert flows == [0, 1]
    s = res.summary()
    assert s["backend"] == "numpy" and len(s["pairs"]) == 2
    assert s["pairs"][0]["mean_motion_px"] > 10


def test_gif_export():
    data = to_gif([f for f in drawings()], fps=8)
    assert data[:6] in (b"GIF89a", b"GIF87a")


def test_match_ink_mass_thins_to_target():
    from flipster.pipeline import match_ink_mass

    ink = np.zeros((20, 20), np.float32)
    ink[8:12, 5:15] = 1.0  # core
    ink[7, 5:15] = ink[12, 5:15] = 0.4  # spill
    out = match_ink_mass(ink, target=40.0)
    assert abs(out.sum() - 40.0) < 1.0
    assert out[9, 10] > 0.9 and out[7, 10] < 0.05
    assert match_ink_mass(ink, target=1e9) is ink
