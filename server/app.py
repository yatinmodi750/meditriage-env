"""
server/app.py — OpenEnv-compliant FastAPI server for MediTriage-Env.
This is the required entry point for openenv validate.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn

from meditriage_env import MediTriageEnv

app = FastAPI(
    title       = "MediTriage-Env",
    description = "OpenEnv-compatible medical triage environment",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_env: MediTriageEnv = MediTriageEnv(task="medium")
_last_obs: Optional[np.ndarray] = None


def obs_to_list(obs: np.ndarray) -> list:
    return [round(float(x), 6) for x in obs]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "env": "MediTriage-Env", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: Request):
    global _env, _last_obs
    try:
        body = await request.json()
    except Exception:
        body = {}

    seed = body.get("seed", None)
    task = body.get("task", "medium")
    if task not in ("easy", "medium", "hard"):
        task = "medium"
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            seed = None

    _env      = MediTriageEnv(task=task)
    _last_obs = _env.reset(seed=seed)

    return JSONResponse({
        "observation": obs_to_list(_last_obs),
        "state":       _env.state(),
        "task":        task,
        "obs_dim":     int(_env.obs_dim),
        "n_actions":   int(_env.n_actions),
    })


@app.post("/step")
async def step(request: Request):
    global _last_obs
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    action = body.get("action", None)
    if action is None:
        return JSONResponse({"error": "Missing 'action'"}, status_code=400)
    try:
        action = int(action)
    except Exception:
        return JSONResponse({"error": f"action must be int, got {action}"}, status_code=400)

    if _env._done:
        _last_obs = _env.reset()

    if not (0 <= action < _env.n_actions):
        return JSONResponse({"error": f"action must be in [0,{_env.n_actions-1}]"}, status_code=400)

    obs, reward, done, info = _env.step(action)
    _last_obs = obs

    return JSONResponse({
        "observation": obs_to_list(obs),
        "reward":      round(float(reward), 6),
        "done":        bool(done),
        "info":        info,
        "state":       _env.state(),
    })


@app.get("/state")
async def state():
    return JSONResponse(_env.state())


@app.get("/")
async def root():
    return JSONResponse({
        "name":              "MediTriage-Env",
        "version":           "1.0.0",
        "description":       "OpenEnv medical triage environment",
        "action_space":      {"type": "Discrete", "n": 16},
        "observation_space": {"type": "Box", "shape": [29], "dtype": "float32"},
        "tasks":             ["easy", "medium", "hard"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)