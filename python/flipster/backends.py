"""Backend discovery and selection.

Three interchangeable engines share one interface
(``flow`` / ``warp`` / ``set_pair`` / ``synthesize`` / ``last_timings``):

* ``numpy`` - the pure-Python reference (always available, slow, used as ground truth)
* ``cpu``   - C++17 + OpenMP (``flipster._core``)
* ``cuda``  - CUDA kernels (``flipster._core`` built with ``FLIPSTER_ENABLE_CUDA``
               and an NVIDIA GPU present)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np

from .params import FlowParams, SplatParams
from .reference import NumpyEngine

try:  # the native extension is optional so the package works from a plain checkout
    from . import _core  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised when the extension is not built
    _core = None


class NativeEngine:
    """Thin adapter that converts dataclass params for the pybind11 engine."""

    def __init__(self, backend: str, variant: str = "optimized") -> None:
        if _core is None:
            raise RuntimeError("flipster._core is not built; `pip install .` to compile it")
        self._e = _core.make_engine(backend, variant)
        self.name = backend if variant == "optimized" else f"{backend}-{variant}"

    @staticmethod
    def _flow_params(p: FlowParams | None):
        p = p or FlowParams()
        cp = _core.FlowParams()
        for k, v in p.to_dict().items():
            setattr(cp, k, v)
        return cp

    @staticmethod
    def _splat_params(p: SplatParams | None):
        p = p or SplatParams()
        cp = _core.SplatParams()
        for k, v in p.to_dict().items():
            setattr(cp, k, v)
        return cp

    def flow(self, a, b, params: FlowParams | None = None) -> np.ndarray:
        return self._e.flow(
            np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32), self._flow_params(params)
        )

    def warp(self, img, flow) -> np.ndarray:
        img = np.ascontiguousarray(img, np.float32)
        squeeze = img.ndim == 2
        out = self._e.warp(img[..., None] if squeeze else img, np.ascontiguousarray(flow, np.float32))
        return out[..., 0] if squeeze else out

    def set_pair(self, c0, c1, imp0, imp1, f01, f10, sp: SplatParams | None = None) -> None:
        c0 = np.ascontiguousarray(c0, np.float32)
        c1 = np.ascontiguousarray(c1, np.float32)
        if c0.ndim == 2:
            c0, c1 = c0[..., None], c1[..., None]
        h, w = c0.shape[:2]
        z = np.zeros((h, w), np.float32)
        imp0 = z if imp0 is None else np.ascontiguousarray(imp0, np.float32)
        imp1 = z if imp1 is None else np.ascontiguousarray(imp1, np.float32)
        self._e.set_pair(
            c0,
            c1,
            imp0,
            imp1,
            np.ascontiguousarray(f01, np.float32),
            np.ascontiguousarray(f10, np.float32),
            self._splat_params(sp),
        )

    def reliability(self):
        return self._e.reliability()

    def synthesize(self, t: float) -> np.ndarray:
        return self._e.synthesize(float(t))

    @property
    def last_timings(self) -> dict[str, float]:
        return dict(self._e.timings())


@lru_cache(maxsize=1)
def available_backends() -> tuple[str, ...]:
    names = []
    if _core is not None:
        if _core.cuda_available():
            names.append("cuda")
        names.append("cpu")
    names.append("numpy")
    return tuple(names)


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"backends": list(available_backends()), "cuda_device": None, "cpu_threads": os.cpu_count()}
    if _core is not None:
        info["cuda_compiled"] = bool(_core.cuda_compiled())
        if _core.cuda_available():
            info["cuda_device"] = _core.cuda_device_name()
    else:
        info["cuda_compiled"] = False
    return info


def get_engine(name: str = "auto", variant: str = "optimized"):
    if name == "auto":
        name = available_backends()[0]
    if name == "numpy":
        return NumpyEngine()
    if name not in available_backends():
        raise ValueError(f"backend {name!r} is not available (have: {', '.join(available_backends())})")
    return NativeEngine(name, variant)
