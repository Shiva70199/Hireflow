from __future__ import annotations

import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from app.env import HireFlowEnv
from app.models import Reward


class CreateSessionRequest(BaseModel):
    task: str = Field(default="hard", pattern="^(easy|medium|hard)$")
    seed: int = 42


class SessionRecord(BaseModel):
    env: HireFlowEnv
    done: bool = False

    model_config = {"arbitrary_types_allowed": True}


app = FastAPI(title="HireFlow MultiAgent SaaS", version="1.0.0")
_SESSIONS: Dict[str, SessionRecord] = {}
_LOCK = Lock()
API_TOKEN = os.getenv("HIREFLOW_API_TOKEN", "")
_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _check_auth(token: str | None) -> None:
    if API_TOKEN and token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


@app.get("/")
def web_ui() -> FileResponse:
    """Browser dashboard (SaaS product UI)."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "hireflow-multiagent-env"}


@app.post("/v1/sessions")
def create_session(payload: CreateSessionRequest, x_api_token: str | None = Header(default=None)) -> dict:
    _check_auth(x_api_token)
    env = HireFlowEnv(task=payload.task, seed=payload.seed)
    obs = env.reset()
    sid = str(uuid.uuid4())
    with _LOCK:
        _SESSIONS[sid] = SessionRecord(env=env, done=False)
    return {"session_id": sid, "observation": obs.model_dump(), "state": env.state()}

@app.post("/reset")
def openenv_reset(
    body: Dict[str, Any] = Body(default_factory=dict),
    x_api_token: str | None = Header(default=None),
) -> JSONResponse:
    """
    OpenEnv-style endpoint for Spaces compatibility.
    Creates a new session and returns the initial observation.
    """
    _check_auth(x_api_token)
    task = str(body.get("task", "hard"))
    seed = int(body.get("seed", 42))
    env = HireFlowEnv(task=task, seed=seed)
    obs = env.reset()
    sid = str(uuid.uuid4())
    with _LOCK:
        _SESSIONS[sid] = SessionRecord(env=env, done=False)
    return JSONResponse(status_code=200, content={"session_id": sid, "observation": obs.model_dump(), "state": env.state()})


@app.post("/v1/sessions/{session_id}/step")
def step_session(
    session_id: str,
    body: Dict[str, Any] = Body(default_factory=dict),
    x_api_token: str | None = Header(default=None),
):
    _check_auth(x_api_token)
    with _LOCK:
        rec = _SESSIONS.get(session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found")
    if rec.done:
        raise HTTPException(status_code=409, detail="Session already completed")

    try:
        obs, reward, done, info = rec.env.step(body)
    except Exception as exc:
        z = Reward(
            total=0.0,
            screening_score=0.0,
            interview_score=0.0,
            final_decision_score=0.0,
            penalties=0.0,
        )
        payload = {
            "observation": {},
            "reward": z.model_dump(),
            "done": rec.done,
            "info": {"error": "step_failed", "message": str(exc)},
            "state": rec.env.state(),
        }
        return JSONResponse(status_code=200, content=payload)

    rec.done = done
    with _LOCK:
        _SESSIONS[session_id] = rec
    return JSONResponse(
        status_code=200,
        content={
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
            "state": rec.env.state(),
        },
    )

@app.post("/step")
def openenv_step(
    body: Dict[str, Any] = Body(default_factory=dict),
    x_api_token: str | None = Header(default=None),
) -> JSONResponse:
    """
    OpenEnv-style endpoint for Spaces compatibility.
    Expects {session_id: str, action: dict} or {session_id: str, ...action_fields}.
    Always returns HTTP 200 with {observation, reward, done, info}.
    """
    _check_auth(x_api_token)
    session_id = body.get("session_id")
    if not session_id:
        z = Reward(total=0.0, screening_score=0.0, interview_score=0.0, final_decision_score=0.0, penalties=0.0)
        return JSONResponse(
            status_code=200,
            content={"observation": {}, "reward": z.model_dump(), "done": False, "info": {"error": "missing_session_id"}, "state": {}},
        )

    with _LOCK:
        rec = _SESSIONS.get(str(session_id))
    if not rec:
        z = Reward(total=0.0, screening_score=0.0, interview_score=0.0, final_decision_score=0.0, penalties=0.0)
        return JSONResponse(
            status_code=200,
            content={"observation": {}, "reward": z.model_dump(), "done": False, "info": {"error": "session_not_found"}, "state": {}},
        )

    action = body.get("action")
    if not isinstance(action, dict):
        action = {k: v for k, v in body.items() if k != "session_id"}

    obs, reward, done, info = rec.env.step(action)
    rec.done = done
    with _LOCK:
        _SESSIONS[str(session_id)] = rec
    return JSONResponse(
        status_code=200,
        content={
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
            "state": rec.env.state(),
        },
    )


@app.get("/v1/sessions/{session_id}/state")
def state_session(session_id: str, x_api_token: str | None = Header(default=None)) -> dict:
    _check_auth(x_api_token)
    with _LOCK:
        rec = _SESSIONS.get(session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"done": rec.done, "state": rec.env.state()}


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str, x_api_token: str | None = Header(default=None)) -> dict:
    _check_auth(x_api_token)
    with _LOCK:
        existed = session_id in _SESSIONS
        if existed:
            del _SESSIONS[session_id]
    return {"deleted": existed}


if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
