import json
import time

import pytest


def make_project(client, pngs, source="drawing"):
    pid = client.post("/api/projects", json={"source": source}).json()["id"]
    files = [("files", (name, data, "image/png")) for name, data in pngs]
    r = client.post(f"/api/projects/{pid}/frames", files=files)
    assert r.status_code == 200, r.text
    return pid, r.json()


def wait_for(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise TimeoutError


def test_health_and_system(client):
    assert client.get("/api/health").json()["status"] == "ok"
    sysinfo = client.get("/api/system").json()
    assert "numpy" in sysinfo["backends"]
    assert sysinfo["limits"]["max_frames"] == 6


def test_upload_reorder_delete(client, drawing_pngs):
    pid, proj = make_project(client, drawing_pngs)
    ids = [f["id"] for f in proj["frames"]]
    assert len(ids) == 3
    assert client.get(proj["frames"][0]["thumb_url"]).headers["content-type"] == "image/jpeg"
    r = client.put(f"/api/projects/{pid}/frames/order", json={"order": ids[::-1]})
    assert [f["id"] for f in r.json()["frames"]] == ids[::-1]
    assert client.put(f"/api/projects/{pid}/frames/order", json={"order": ids[:2]}).status_code == 400
    r = client.delete(f"/api/projects/{pid}/frames/{ids[1]}")
    assert [f["id"] for f in r.json()["frames"]] == [ids[2], ids[0]]


def test_rejects_bad_uploads_and_ids(client, drawing_pngs):
    pid = client.post("/api/projects", json={"source": "scan"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/frames", files=[("files", ("x.png", b"not an image", "image/png"))])
    assert r.status_code == 400
    assert client.get("/api/projects/../../etc").status_code == 404
    assert client.get("/api/projects/zzzzzzzzzzzz").status_code == 404
    # frame limit
    files = [("files", (n, d, "image/png")) for n, d in drawing_pngs * 3]
    assert client.post(f"/api/projects/{pid}/frames", files=files).status_code == 400


def test_render_flow_end_to_end(client, drawing_pngs):
    pid, _ = make_project(client, drawing_pngs)
    r = client.post(f"/api/projects/{pid}/renders", json={"method": "flow", "inbetweens": 2, "backend": "numpy"})
    assert r.status_code == 202, r.text
    job = wait_for(client, r.json()["job_id"])
    assert job["status"] == "done", job
    render = client.get(f"/api/renders/{r.json()['render_id']}").json()
    assert len(render["frames"]) == 3 + 2 * 2
    assert [f["key"] for f in render["frames"]] == [True, False, False, True, False, False, True]
    assert len(render["flow_urls"]) == 2
    assert render["summary"]["backend"] == "numpy"
    img = client.get(render["frames"][1]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    gif = client.get(f"/api/renders/{render['id']}/export?format=gif&fps=10&pingpong=true")
    assert gif.status_code == 200 and gif.content[:3] == b"GIF"


def test_render_events_stream(client, drawing_pngs):
    pid, _ = make_project(client, drawing_pngs)
    r = client.post(f"/api/projects/{pid}/renders", json={"method": "linear", "inbetweens": 1})
    events = []
    with client.stream("GET", f"/api/jobs/{r.json()['job_id']}/events") as s:
        for line in s.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert events[-1]["status"] == "done"
    progress = [e["progress"] for e in events]
    assert progress == sorted(progress)


@pytest.mark.parametrize("body", [{"inbetweens": 99}, {"method": "magic"}, {"backend": "quantum"}])
def test_render_validation(client, drawing_pngs, body):
    pid, _ = make_project(client, drawing_pngs[:2])
    assert client.post(f"/api/projects/{pid}/renders", json=body).status_code in (400, 422)


def test_render_requires_frames(client):
    pid = client.post("/api/projects", json={}).json()["id"]
    assert client.post(f"/api/projects/{pid}/renders", json={}).status_code == 400
