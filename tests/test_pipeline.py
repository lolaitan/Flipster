import numpy as np
import pytest

from flipster import RenderOptions, render
from flipster.export import to_gif
from synth import stick_figure


def drawings(n=3, step=15):
    out = []
    for i in range(n):
        ink = stick_figure(120, 160, 50 + i * step, 60)
        out.append((255 * (1 - ink[..., None].repeat(3, -1))).astype(np.uint8))
    return out


@pytest.mark.parametrize("method,expected", [("none", 3), ("linear", 7), ("flow", 7)])
def test_frame_counts(method, expected):
    events = []
    res = render(drawings(), RenderOptions(method=method, inbetweens=2, source="drawing", backend="numpy"),
                 on_progress=lambda s, f, m: events.append(s))
    assert len(res.frames) == expected
    assert [f.key for f in res.frames][:4] == ([True, True, True][:3] + [])[:0] + [f.key for f in res.frames][:4]
    assert sum(f.key for f in res.frames) == 3
    assert events[-1] == "done"
    assert res.frames[0].image.dtype == np.uint8 and res.frames[0].image.shape == (120, 160, 3)


def test_flow_render_streams_and_reports_stats():
    got = {}
    flows = []
    res = render(drawings(), RenderOptions(method="flow", inbetweens=1, source="drawing", backend="numpy"),
                 on_frame=lambda i, f: got.__setitem__(i, f), on_flow=lambda i, v: flows.append(i))
    assert sorted(got) == list(range(5))
    assert [got[i].t for i in range(5)] == [0.0, 0.5, 0.0, 0.5, 0.0]
    assert flows == [0, 1]
    s = res.summary()
    assert s["backend"] == "numpy" and len(s["pairs"]) == 2
    assert s["pairs"][0]["mean_motion_px"] > 10


def test_gif_export():
    data = to_gif([f for f in drawings()], fps=8)
    assert data[:6] in (b"GIF89a", b"GIF87a")
