"""Parameter objects shared by every backend (NumPy reference, C++ CPU, CUDA)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class FlowParams:
    """Dense pyramidal Lucas-Kanade (Bouguet-style coarse-to-fine, iterative warping).

    levels:        pyramid levels; 0 = choose automatically from image size.
    max_levels:    upper bound when ``levels == 0``.
    min_size:      coarsest level keeps min(h, w) >= min_size.
    iterations:    Gauss-Newton iterations per level.
    window_radius: LK window is (2r+1) x (2r+1).
    damping:       Levenberg-Marquardt damping toward the current estimate, relative to
                   the window's gradient energy ((Gxx + Gyy) / 2), so every pyramid
                   level converges at the same rate.
    damping_floor: absolute damping added on top; keeps the 2x2 solve stable in
                   textureless windows, which then inherit the flow propagated from
                   coarser levels.
    zero_pull:     weak prior toward zero motion. Lets directions the window cannot
                   observe (the aperture problem, e.g. along a straight stroke)
                   relax to "static" instead of inheriting blurred coarse motion.
    max_step:      per-iteration update clamp in pixels (robustness).
    median:        3x3 median filter on the flow after each level.
    """

    levels: int = 0
    max_levels: int = 6
    min_size: int = 16
    iterations: int = 5
    window_radius: int = 7
    damping: float = 0.05
    damping_floor: float = 1e-6
    zero_pull: float = 0.0
    max_step: float = 4.0
    median: bool = True

    def resolve_levels(self, height: int, width: int) -> int:
        if self.levels > 0:
            return self.levels
        levels, h, w = 1, height, width
        while levels < self.max_levels and min((h + 1) // 2, (w + 1) // 2) >= self.min_size:
            h, w = (h + 1) // 2, (w + 1) // 2
            levels += 1
        return levels

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplatParams:
    """Occlusion-aware forward splatting used to synthesize in-between frames.

    softmax_beta: importance weight is exp(beta * importance) (softmax splatting,
                  Niklaus & Liu 2020). For line art the importance is the ink map, so
                  strokes win over blank paper when both land on the same pixel.
    fb_alpha/fb_beta: forward-backward consistency threshold
                  |F01 + F10(x+F01)|^2 <= alpha (|F01|^2 + |F10|^2) + beta  (Sundaram et al. 2010).
    min_reliability: floor so a pixel is never fully discarded.
    """

    softmax_beta: float = 10.0
    fb_alpha: float = 0.05
    fb_beta: float = 1.0
    min_reliability: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


Method = Literal["none", "linear", "flow"]
Style = Literal["clean", "photo"]
Source = Literal["scan", "drawing", "photo"]


@dataclass
class RenderOptions:
    method: Method = "flow"
    inbetweens: int = 3
    style: Style = "clean"
    source: Source = "scan"
    backend: str = "auto"
    working_height: int = 1200
    register: bool = True
    flow: FlowParams = field(default_factory=FlowParams)
    splat: SplatParams = field(default_factory=SplatParams)
