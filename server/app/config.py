"""Runtime configuration (environment variables with sensible defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    samples_dir: Path
    web_dist: Path
    max_frames: int
    max_upload_mb: int
    max_working_height: int
    ttl_hours: float
    cors_origins: tuple[str, ...]


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        data_dir=Path(env("FLIPSTER_DATA_DIR", REPO_ROOT / "data")),
        samples_dir=Path(env("FLIPSTER_SAMPLES_DIR", REPO_ROOT / "samples" / "scans")),
        web_dist=Path(env("FLIPSTER_WEB_DIST", REPO_ROOT / "web" / "dist")),
        max_frames=int(env("FLIPSTER_MAX_FRAMES", "60")),
        max_upload_mb=int(env("FLIPSTER_MAX_UPLOAD_MB", "25")),
        max_working_height=int(env("FLIPSTER_MAX_WORKING_HEIGHT", "1600")),
        ttl_hours=float(env("FLIPSTER_TTL_HOURS", "24")),
        cors_origins=tuple(o for o in env("FLIPSTER_CORS_ORIGINS", "http://localhost:5173").split(",") if o),
    )
