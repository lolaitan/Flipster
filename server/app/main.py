"""Flipster HTTP API.

Run locally:  uvicorn app.main:app --app-dir server --reload
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import flipster
from flipster.export import to_gif, to_mp4

from .config import Settings, load_settings
from .jobs import JobManager
from .schemas import (
    CreateProject,
    CreateRenderOut,
    FrameOut,
    JobOut,
    ProjectOut,
    RenderOut,
    RenderRequest,
    ReorderFrames,
)
from .storage import BadRequest, NotFound, Storage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    storage = Storage(settings.data_dir)
    jobs = JobManager(storage, settings.max_working_height)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        storage.purge_older_than(settings.ttl_hours)
        yield
        jobs.shutdown()

    app = FastAPI(title="Flipster", version=flipster.__version__, lifespan=lifespan)
    app.state.settings, app.state.storage, app.state.jobs = settings, storage, jobs
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound):
        return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)

    @app.exception_handler(BadRequest)
    async def _bad_request(_: Request, exc: BadRequest):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ------------------------------------------------------------ helpers

    def project_out(meta: dict) -> ProjectOut:
        pid = meta["id"]
        return ProjectOut(
            id=pid,
            source=meta["source"],
            created_at=meta["created_at"],
            frames=[
                FrameOut(
                    id=f["id"],
                    name=f["name"],
                    width=f["width"],
                    height=f["height"],
                    image_url=f"/api/projects/{pid}/frames/{f['id']}/image",
                    thumb_url=f"/api/projects/{pid}/frames/{f['id']}/thumb",
                )
                for f in meta["frames"]
            ],
        )

    def render_out(meta: dict) -> RenderOut:
        rid = meta["id"]
        return RenderOut(
            id=rid,
            project_id=meta["project_id"],
            status=meta["status"],
            options=meta["options"],
            summary=meta.get("summary"),
            frames=[{**f, "url": f"/api/renders/{rid}/frames/{f['index']}.jpg"} for f in meta.get("frames", [])],
            flow_urls=[f"/api/renders/{rid}/flow/{i}.jpg" for i in range(meta.get("flows", 0))],
        )

    cache = {"Cache-Control": "public, max-age=31536000, immutable"}

    # ------------------------------------------------------------ system

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": flipster.__version__}

    @app.get("/api/system")
    def system():
        return {
            **flipster.device_info(),
            "limits": {
                "max_frames": settings.max_frames,
                "max_upload_mb": settings.max_upload_mb,
                "max_working_height": settings.max_working_height,
            },
            "samples": settings.samples_dir.exists(),
        }

    # ------------------------------------------------------------ projects

    @app.post("/api/projects", response_model=ProjectOut, status_code=201)
    def create_project(body: CreateProject):
        return project_out(storage.create_project(body.source))

    @app.get("/api/projects/{pid}", response_model=ProjectOut)
    def get_project(pid: str):
        return project_out(storage.get_project(pid))

    @app.post("/api/projects/{pid}/frames", response_model=ProjectOut)
    async def upload_frames(pid: str, files: list[UploadFile] = File(...)):
        storage.get_project(pid)
        limit = settings.max_upload_mb * 1024 * 1024
        meta = None
        for f in files:
            data = await f.read(limit + 1)
            if len(data) > limit:
                raise BadRequest(f"{f.filename}: larger than {settings.max_upload_mb} MB")
            meta = await asyncio.to_thread(storage.add_frame, pid, data, f.filename or "frame", settings.max_frames)
        return project_out(meta or storage.get_project(pid))

    @app.post("/api/projects/{pid}/samples", response_model=ProjectOut)
    def load_samples(pid: str):
        if not settings.samples_dir.exists():
            raise NotFound("samples")
        meta = storage.get_project(pid)
        for p in sorted(settings.samples_dir.glob("*.jpg")):
            meta = storage.add_frame(pid, p.read_bytes(), p.name, settings.max_frames)
        return project_out(meta)

    @app.put("/api/projects/{pid}/frames/order", response_model=ProjectOut)
    def reorder(pid: str, body: ReorderFrames):
        return project_out(storage.reorder(pid, body.order))

    @app.delete("/api/projects/{pid}/frames/{fid}", response_model=ProjectOut)
    def delete_frame(pid: str, fid: str):
        return project_out(storage.delete_frame(pid, fid))

    @app.get("/api/projects/{pid}/frames/{fid}/image")
    def frame_image(pid: str, fid: str):
        return FileResponse(storage.frame_path(pid, fid), headers=cache)

    @app.get("/api/projects/{pid}/frames/{fid}/thumb")
    def frame_thumb(pid: str, fid: str):
        return FileResponse(storage.frame_path(pid, fid, thumb=True), headers=cache)

    # ------------------------------------------------------------ renders

    @app.post("/api/projects/{pid}/renders", response_model=CreateRenderOut, status_code=202)
    def create_render(pid: str, body: RenderRequest):
        meta = storage.get_project(pid)
        if not meta["frames"]:
            raise BadRequest("add at least one frame first")
        if body.backend != "auto" and body.backend not in flipster.available_backends():
            raise BadRequest(f"backend {body.backend!r} not available on this server")
        job = jobs.submit(pid, meta["source"], body.model_dump())
        return CreateRenderOut(job_id=job.id, render_id=job.render_id)

    @app.get("/api/jobs/{jid}", response_model=JobOut)
    def get_job(jid: str):
        snap = jobs.snapshot(jid)
        if not snap:
            raise NotFound(jid)
        return snap[1]

    @app.get("/api/jobs/{jid}/events")
    async def job_events(jid: str, request: Request):
        """Server-sent events: one message per job state change until it finishes."""
        if not jobs.snapshot(jid):
            raise NotFound(jid)

        async def stream():
            last = -1
            while True:
                if await request.is_disconnected():
                    return
                version, state = jobs.snapshot(jid)
                if version != last:
                    last = version
                    yield f"data: {json.dumps(state)}\n\n"
                    if state["status"] in ("done", "error"):
                        return
                await asyncio.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/renders/{rid}", response_model=RenderOut)
    def get_render(rid: str):
        return render_out(storage.get_render(rid))

    @app.get("/api/renders/{rid}/frames/{index}.jpg")
    def render_frame(rid: str, index: int):
        return FileResponse(storage.render_frame_path(rid, index), headers=cache)

    @app.get("/api/renders/{rid}/flow/{pair}.jpg")
    def render_flow(rid: str, pair: int):
        return FileResponse(storage.flow_path(rid, pair), headers=cache)

    @app.get("/api/renders/{rid}/export")
    async def export(
        rid: str,
        format: str = Query("gif", pattern="^(gif|mp4)$"),
        fps: float = Query(12, gt=0, le=60),
        max_side: int = Query(720, ge=64, le=2160),
        pingpong: bool = False,
    ):
        meta = storage.get_render(rid)
        if meta["status"] != "done":
            raise BadRequest("render is not finished")
        paths = [storage.render_frame_path(rid, f["index"]) for f in meta["frames"]]
        if pingpong and len(paths) > 2:
            paths = paths + paths[-2:0:-1]

        def build() -> bytes:
            import numpy as np

            frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
            return to_gif(frames, fps, max_side) if format == "gif" else to_mp4(frames, fps, max_side)

        data = await asyncio.to_thread(build)
        media = "image/gif" if format == "gif" else "video/mp4"
        return Response(
            data, media_type=media, headers={"Content-Disposition": f'attachment; filename="flipster-{rid}.{format}"'}
        )

    # ------------------------------------------------------------ frontend

    dist: Path = settings.web_dist
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app


app = create_app()
