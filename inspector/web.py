"""Web app + public MCP endpoint, one process.

  /            inspection console (static/index.html)
  /api/...     panels, runs (live trace over SSE), approvals, tickets
  /mcp         the same OpenCV 5 tools over MCP (Streamable HTTP) for any agent

Run: uvicorn inspector.web:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, telemetry
from . import vision as v
from .server import make_demo_panel, mcp, opencv_info
from .store import STORE

STATIC = Path(__file__).with_name("static")
MAX_UPLOAD = 15 * 1024 * 1024
RUNS: dict[str, dict] = {}
RUN_LOCK = asyncio.Lock()  # one inspection at a time keeps the small server responsive


def _gemini_ready() -> bool:
    from .llm import _load_env
    _load_env()
    return bool(os.environ.get("GEMINI_API_KEY"))


mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp", host="0.0.0.0")


@contextlib.asynccontextmanager
async def lifespan(_app):
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="InspectorMCP", lifespan=lifespan)


# ---------- pages & info ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "time": time.time()}


@app.get("/api/info")
def info():
    return {**opencv_info(), "gemini": _gemini_ready(), "mcp_endpoint": "/mcp"}


# ---------- panels & images ----------

class DemoReq(BaseModel):
    rows: int = 4
    cols: int = 6
    seed: int | None = None
    defect_share: float = 0.25


@app.post("/api/panels/demo")
def new_demo(req: DemoReq):
    if not (1 <= req.rows <= 8 and 1 <= req.cols <= 12):
        raise HTTPException(400, "rows 1-8, cols 1-12")
    seed = req.seed if req.seed is not None else int(time.time()) % 100000
    return {**make_demo_panel(req.rows, req.cols, req.defect_share, seed), "seed": seed}


@app.post("/api/panels/upload")
async def upload(file: UploadFile = File(...), rows: int = Form(...), cols: int = Form(...)):
    if not (1 <= rows <= 12 and 1 <= cols <= 24):
        raise HTTPException(400, "rows 1-12, cols 1-24")
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (15 MB max)")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "not an image")
    if max(img.shape) > 4000:  # keep memory and time bounded
        s = 4000 / max(img.shape)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    image_id = STORE.put(img, "panel", rows=rows, cols=cols, source="upload")
    return {"image_id": image_id, "rows": rows, "cols": cols, "size": list(img.shape)}


@app.get("/api/images/{image_id}.png")
def image(image_id: str, overlay: bool = False):
    try:
        g = STORE.get(image_id)
    except ValueError:
        raise HTTPException(404, "unknown image")
    if overlay:
        g = v.draw_findings(g, v.find_cracks(g), v.find_dark_areas(g))
    ok, buf = cv2.imencode(".png", g)
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# ---------- runs ----------

class RunReq(BaseModel):
    panel_id: str
    rows: int
    cols: int
    brain: str = "policy"  # policy | gemini


async def _run(run_id: str, req: RunReq):
    run = RUNS[run_id]

    def on_step(step: dict):
        run["events"].append(step)
        run["wake"].set()

    try:
        if req.brain == "gemini":
            from .llm import GeminiBrain
            brain = GeminiBrain()
        else:
            brain = agent.PolicyBrain()
        t0 = time.perf_counter()
        async with RUN_LOCK:
            out = await agent.inspect_panel(mcp, req.panel_id, req.rows, req.cols, brain, on_step)
        secs = time.perf_counter() - t0
        decisions = [r["decision"] for r in out["results"]]
        truth = STORE.meta.get(req.panel_id, {}).get("truth")
        summary = {
            "run_id": run_id, "panel": req.panel_id, "brain": brain.name, "cells": len(decisions),
            "pass": decisions.count("pass"), "fail": decisions.count("fail"), "human": decisions.count("human"),
            "seconds": round(secs, 2), "ms_per_cell": round(1000 * secs / max(len(decisions), 1), 1),
            "tool_calls": sum(1 for s in out["trace"] if s["kind"] == "tool"),
            "llm_requests": getattr(brain, "requests", 0),
        }
        if truth:  # demo panels know the real answer - show how the agent did
            auto = [(d, t) for d, t in zip(decisions, truth) if d != "human"]
            summary["agreement_with_labels"] = round(sum((d == "fail") == bool(t) for d, t in auto) / max(len(auto), 1), 3)
        run.update(results=out["results"], summary=summary, status="done")
        telemetry.report_run(summary)
    except Exception as e:
        run.update(status="error", error=str(e)[:500])
    finally:
        run["events"].append({"kind": "end", "status": run["status"]})
        run["wake"].set()


@app.post("/api/runs")
async def start_run(req: RunReq):
    if req.brain not in ("policy", "gemini"):
        raise HTTPException(400, "brain must be policy or gemini")
    if req.brain == "gemini" and not _gemini_ready():
        raise HTTPException(400, "Gemini key not configured on this server")
    if RUN_LOCK.locked():
        raise HTTPException(429, "another inspection is running - try again in a moment")
    STORE.get(req.panel_id)  # 404-style check early
    run_id = uuid.uuid4().hex[:10]
    RUNS[run_id] = {"id": run_id, "status": "running", "events": [], "wake": asyncio.Event(),
                    "request": req.model_dump(), "started": time.time()}
    asyncio.create_task(_run(run_id, req))
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    return {k: val for k, val in run.items() if k != "wake"}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")

    async def stream():
        i = 0
        while True:
            while i < len(run["events"]):
                yield f"data: {json.dumps(run['events'][i])}\n\n"
                i += 1
            if run["status"] != "running" and i >= len(run["events"]):
                return
            run["wake"].clear()
            try:
                await asyncio.wait_for(run["wake"].wait(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------- human in the loop ----------

@app.get("/api/approvals")
def approvals():
    return sorted(STORE.approvals.values(), key=lambda a: a["created"], reverse=True)[:100]


class Decision(BaseModel):
    defect: bool  # the human's answer: is this cell defective?
    note: str = ""


@app.post("/api/approvals/{approval_id}")
def decide(approval_id: str, d: Decision):
    if approval_id not in STORE.approvals:
        raise HTTPException(404, "unknown approval")
    if STORE.approvals[approval_id]["status"] != "pending":
        raise HTTPException(409, "already decided")
    req = STORE.decide(approval_id, d.defect, d.note[:500])
    # the human's answer becomes the final verdict for that cell
    if d.defect:
        STORE.add_ticket(image_id=req["image_id"], panel=req.get("panel"), row=req.get("row"), col=req.get("col"),
                         defect="confirmed by human", severity="medium",
                         evidence=f"human review: {d.note or 'defect confirmed'}; agent: {req.get('reason', '')}")
    else:
        STORE.set_verdict(req["image_id"], "pass", f"human review: {d.note or 'no defect'}")
    return {**req, "final": "fail" if d.defect else "pass"}


@app.get("/api/tickets")
def tickets():
    return list(reversed(STORE.tickets[-100:]))


app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")
# MCP (Streamable HTTP) at /mcp - mounted last so the routes above win
app.mount("/", mcp_app)
