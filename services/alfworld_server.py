from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


DATA_ROOT = Path(os.environ.get("ALFWORLD_GAMES_ROOT", "/data/games/games")).resolve()
app = FastAPI()
sessions = {}
textworld_lock = threading.Lock()


class ResetRequest(BaseModel):
    game_relative_path: str


class StepRequest(BaseModel):
    session_id: str
    action: str


class CloseRequest(BaseModel):
    session_id: str


def state_payload(session_id: str, state, reward: float, done: bool) -> dict:
    return {
        "session_id": session_id,
        "observation": state.feedback,
        "admissible_actions": list(state.admissible_commands),
        "reward": float(reward),
        "won": bool(state.won),
        "done": bool(done),
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/reset")
def reset(request: ResetRequest) -> dict:
    game_path = (DATA_ROOT / request.game_relative_path).resolve()
    if os.path.commonpath((DATA_ROOT, game_path)) != str(DATA_ROOT) or not game_path.is_file():
        raise HTTPException(status_code=400, detail="invalid game path")

    infos = textworld.EnvInfos(won=True, admissible_commands=True)
    # TextWorld 的模块级 Tatsu parser 不能被多个环境线程同时调用。
    with textworld_lock:
        environment = textworld.start(
            str(game_path),
            infos=infos,
            wrappers=[AlfredDemangler(shuffle=False)],
        )
        state = environment.reset()
    session_id = uuid.uuid4().hex
    sessions[session_id] = environment
    return state_payload(session_id, state, 0, False)


@app.post("/step")
def step(request: StepRequest) -> dict:
    environment = sessions.get(request.session_id)
    if environment is None:
        raise HTTPException(status_code=404, detail="unknown session")
    with textworld_lock:
        state, reward, done = environment.step(request.action)
    return state_payload(request.session_id, state, reward, done)


@app.post("/close")
def close(request: CloseRequest) -> dict:
    environment = sessions.pop(request.session_id, None)
    if environment is not None:
        environment.close()
    return {"closed": environment is not None}
