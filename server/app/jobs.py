"""Background render jobs.

Renders run on a single worker thread: the GPU (or all CPU cores) is the scarce
resource, so jobs queue instead of competing. State lives in memory; results are
on disk, so a restart loses only in-flight jobs.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from flipster import FlowParams, RenderOptions, render
from flipster.preprocess import load_rgb

from .storage import Storage, new_id

# Share of the progress bar given to each pipeline stage.
_STAGE_SPAN = {"preprocess": (0.0, 0.1), "interpolate": (0.1, 1.0), "done": (1.0, 1.0)}


@dataclass
class Job:
    id: str
    render_id: str
    project_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting for a worker"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    version: int = 0

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("version")
        return d


class JobManager:
    def __init__(self, storage: Storage, max_working_height: int) -> None:
        self.storage = storage
        self.max_working_height = max_working_height
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> tuple[int, dict[str, Any]] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return (job.version, job.public()) if job else None

    def _update(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)
            job.version += 1

    def submit(self, project_id: str, source: str, request: dict[str, Any]) -> Job:
        render_id = self.storage.create_render(project_id, request)
        job = Job(id=new_id(), render_id=render_id, project_id=project_id)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, source, request)
        return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ worker

    def _run(self, job: Job, source: str, req: dict[str, Any]) -> None:
        self._update(job, status="running", stage="loading", message="Loading pages", started_at=time.time())
        self.storage.update_render(job.render_id, status="running")
        try:
            paths = self.storage.ordered_frame_paths(job.project_id)
            if len(paths) < 1:
                raise ValueError("add at least one frame first")
            images = [load_rgb(str(p)) for p in paths]
            opts = RenderOptions(
                method=req["method"],
                inbetweens=req["inbetweens"],
                style=req["style"] if source != "drawing" else "clean",
                source=source,
                backend=req["backend"],
                working_height=min(req["working_height"], self.max_working_height),
                register=req["stabilize"],
                flow=FlowParams(**req["flow"]),
            )
            rdir = self.storage.rdir(job.render_id)
            frames: list[dict[str, Any]] = []
            flows: list[int] = []

            def on_frame(i: int, f) -> None:
                Image.fromarray(f.image).save(rdir / "frames" / f"{i:04d}.jpg", quality=90)
                frames.append({"index": i, "key": f.key, "source": f.source, "t": round(f.t, 4)})

            def on_flow(i: int, img: np.ndarray) -> None:
                Image.fromarray(img).save(rdir / "flow" / f"{i:03d}.jpg", quality=85)
                flows.append(i)

            def on_progress(stage: str, frac: float, message: str) -> None:
                lo, hi = _STAGE_SPAN.get(stage, (0.0, 1.0))
                self._update(job, stage=stage, progress=round(lo + (hi - lo) * frac, 4), message=message)

            result = render(images, opts, on_progress=on_progress, on_frame=on_frame, on_flow=on_flow)
            summary = result.summary()
            summary["frames"] = len(frames)
            self.storage.update_render(job.render_id, status="done", summary=summary, frames=frames, flows=len(flows))
            self._update(job, status="done", stage="done", progress=1.0,
                         message=f"{len(frames)} frames in {summary['total_ms'] / 1000:.1f}s on {summary['backend']}",
                         finished_at=time.time())
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            traceback.print_exc()
            self.storage.update_render(job.render_id, status="error")
            self._update(job, status="error", stage="error", error=str(exc), message="Render failed",
                         finished_at=time.time())
