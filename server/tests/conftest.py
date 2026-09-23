import io
import os
import sys
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

from fastapi.testclient import TestClient  # noqa: E402
from synth import stick_figure  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    settings = replace(load_settings(), data_dir=tmp_path / "data", web_dist=tmp_path / "nope", max_frames=6)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture()
def drawing_pngs():
    out = []
    for i in range(3):
        ink = stick_figure(96, 128, 40 + 14 * i, 48)
        rgb = (255 * (1 - np.repeat(ink[..., None], 3, -1))).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        out.append((f"f{i}.png", buf.getvalue()))
    return out
