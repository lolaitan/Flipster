"""API request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FrameOut(BaseModel):
    id: str
    name: str
    width: int
    height: int
    image_url: str
    thumb_url: str


class ProjectOut(BaseModel):
    id: str
    source: Literal["scan", "drawing", "photo"]
    created_at: float
    frames: list[FrameOut]


class CreateProject(BaseModel):
    source: Literal["scan", "drawing", "photo"] = "scan"


class ReorderFrames(BaseModel):
    order: list[str]


class FlowSettings(BaseModel):
    levels: int = Field(0, ge=0, le=8)
    iterations: int = Field(5, ge=1, le=20)
    window_radius: int = Field(7, ge=1, le=15)
    damping: float = Field(0.05, ge=0.0, le=10.0)
    zero_pull: float = Field(0.0, ge=0.0, le=1.0)
    median: bool = True


class RenderRequest(BaseModel):
    method: Literal["none", "linear", "flow"] = "flow"
    inbetweens: int = Field(3, ge=0, le=8)
    style: Literal["clean", "photo"] = "clean"
    backend: str = "auto"
    stabilize: bool = True  # align pages on the printed paper (scans only)
    working_height: int = Field(1200, ge=128, le=4096)
    flow: FlowSettings = FlowSettings()


class JobOut(BaseModel):
    id: str
    render_id: str
    project_id: str
    status: Literal["queued", "running", "done", "error"]
    stage: str
    progress: float
    message: str
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class RenderFrameOut(BaseModel):
    index: int
    key: bool
    source: int
    t: float
    url: str


class RenderOut(BaseModel):
    id: str
    project_id: str
    status: str
    options: dict
    summary: dict | None = None
    frames: list[RenderFrameOut] = []
    flow_urls: list[str] = []


class CreateRenderOut(BaseModel):
    job_id: str
    render_id: str
