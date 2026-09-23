"""On-disk project / render storage.

Layout::

    data/
      projects/<pid>/project.json        metadata + frame order
      projects/<pid>/frames/<fid>.<ext>  uploaded page (original bytes)
      projects/<pid>/frames/<fid>_t.jpg  thumbnail
      renders/<rid>/meta.json
      renders/<rid>/frames/0000.jpg ...
      renders/<rid>/flow/000.jpg ...

Ids are random hex, validated on every access, so client-supplied ids can never
escape the data directory.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

_ID = re.compile(r"^[0-9a-f]{12}$")
THUMB_SIDE = 360
ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tif", "MPO": "jpg"}


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def check_id(value: str) -> str:
    if not _ID.match(value or ""):
        raise NotFound(value)
    return value


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.projects = self.root / "projects"
        self.renders = self.root / "renders"
        self.projects.mkdir(parents=True, exist_ok=True)
        self.renders.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ projects

    def _pdir(self, pid: str) -> Path:
        d = self.projects / check_id(pid)
        if not (d / "project.json").exists():
            raise NotFound(pid)
        return d

    def create_project(self, source: str) -> dict[str, Any]:
        pid = new_id()
        d = self.projects / pid
        (d / "frames").mkdir(parents=True)
        meta = {"id": pid, "source": source, "created_at": time.time(), "frames": []}
        self._write(d / "project.json", meta)
        return meta

    def get_project(self, pid: str) -> dict[str, Any]:
        with self._lock:
            return json.loads((self._pdir(pid) / "project.json").read_text())

    def _save_project(self, meta: dict[str, Any]) -> None:
        self._write(self._pdir(meta["id"]) / "project.json", meta)

    def add_frame(self, pid: str, data: bytes, name: str, max_frames: int) -> dict[str, Any]:
        try:
            im = Image.open(io.BytesIO(data))
            fmt = im.format
            im = ImageOps.exif_transpose(im)
            im.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise BadRequest(f"{name}: not a readable image") from exc
        if fmt not in ALLOWED_FORMATS:
            raise BadRequest(f"{name}: unsupported format {fmt}")
        with self._lock:
            meta = self.get_project(pid)
            if len(meta["frames"]) >= max_frames:
                raise BadRequest(f"a project can hold at most {max_frames} frames")
            fid = new_id()
            d = self._pdir(pid) / "frames"
            path = d / f"{fid}.{ALLOWED_FORMATS[fmt]}"
            path.write_bytes(data)
            thumb = im.convert("RGB")
            thumb.thumbnail((THUMB_SIDE, THUMB_SIDE))
            thumb.save(d / f"{fid}_t.jpg", quality=82)
            meta["frames"].append({"id": fid, "name": name[:120], "file": path.name, "width": im.width, "height": im.height})
            self._save_project(meta)
            return meta

    def frame_path(self, pid: str, fid: str, thumb: bool = False) -> Path:
        meta = self.get_project(pid)
        for f in meta["frames"]:
            if f["id"] == check_id(fid):
                return self._pdir(pid) / "frames" / (f"{fid}_t.jpg" if thumb else f["file"])
        raise NotFound(fid)

    def ordered_frame_paths(self, pid: str) -> list[Path]:
        meta = self.get_project(pid)
        return [self._pdir(pid) / "frames" / f["file"] for f in meta["frames"]]

    def reorder(self, pid: str, order: list[str]) -> dict[str, Any]:
        with self._lock:
            meta = self.get_project(pid)
            by_id = {f["id"]: f for f in meta["frames"]}
            if sorted(order) != sorted(by_id):
                raise BadRequest("order must list every frame id exactly once")
            meta["frames"] = [by_id[i] for i in order]
            self._save_project(meta)
            return meta

    def delete_frame(self, pid: str, fid: str) -> dict[str, Any]:
        with self._lock:
            meta = self.get_project(pid)
            keep = [f for f in meta["frames"] if f["id"] != check_id(fid)]
            if len(keep) == len(meta["frames"]):
                raise NotFound(fid)
            d = self._pdir(pid) / "frames"
            for f in meta["frames"]:
                if f["id"] == fid:
                    (d / f["file"]).unlink(missing_ok=True)
                    (d / f"{fid}_t.jpg").unlink(missing_ok=True)
            meta["frames"] = keep
            self._save_project(meta)
            return meta

    # ------------------------------------------------------------ renders

    def rdir(self, rid: str) -> Path:
        return self.renders / check_id(rid)

    def create_render(self, pid: str, options: dict[str, Any]) -> str:
        rid = new_id()
        d = self.renders / rid
        (d / "frames").mkdir(parents=True)
        (d / "flow").mkdir()
        self._write(d / "meta.json", {"id": rid, "project_id": pid, "status": "queued", "options": options,
                                      "summary": None, "frames": [], "flows": 0})
        return rid

    def get_render(self, rid: str) -> dict[str, Any]:
        p = self.rdir(rid) / "meta.json"
        if not p.exists():
            raise NotFound(rid)
        return json.loads(p.read_text())

    def update_render(self, rid: str, **fields: Any) -> None:
        with self._lock:
            meta = self.get_render(rid)
            meta.update(fields)
            self._write(self.rdir(rid) / "meta.json", meta)

    def render_frame_path(self, rid: str, index: int) -> Path:
        p = self.rdir(rid) / "frames" / f"{index:04d}.jpg"
        if index < 0 or not p.exists():
            raise NotFound(str(index))
        return p

    def flow_path(self, rid: str, pair: int) -> Path:
        p = self.rdir(rid) / "flow" / f"{pair:03d}.jpg"
        if pair < 0 or not p.exists():
            raise NotFound(str(pair))
        return p

    # ------------------------------------------------------------ housekeeping

    def purge_older_than(self, hours: float) -> int:
        cutoff = time.time() - hours * 3600
        removed = 0
        for base in (self.projects, self.renders):
            for d in base.iterdir():
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
        return removed

    @staticmethod
    def _write(path: Path, obj: Any) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(path)
