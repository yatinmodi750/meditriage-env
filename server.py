"""
server.py — FastAPI HTTP server exposing the OpenEnv API.

Endpoints:
    POST /reset   → resets environment, returns initial observation
    POST /step    → takes action, returns obs/reward/done/info
    GET  /state   → returns current environment state
    GET  /health  → health check
    GET  /        → API info
"""
from __future__ import annotations
import os
from typing import Optional, Any
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
    """
    Reset the environment.
    Accepts empty body, {} or {"seed": int, "task": "easy"|"medium"|"hard"}
    """
    global _env, _last_obs

    # Parse body — handle empty body, missing fields gracefully
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
    state     = _env.state()

    return JSONResponse({
        "observation": obs_to_list(_last_obs),
        "state":       state,
        "task":        task,
        "obs_dim":     int(_env.obs_dim),
        "n_actions":   int(_env.n_actions),
    })


@app.post("/step")
async def step(request: Request):
    """
    Take one step. Body: {"action": int}
    """
    global _last_obs

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    action = body.get("action", None)
    if action is None:
        return JSONResponse({"error": "Missing 'action' in request body"}, status_code=400)

    try:
        action = int(action)
    except Exception:
        return JSONResponse({"error": f"action must be an integer, got {action}"}, status_code=400)

    if _env._done:
        # Auto-reset if episode is done
        _last_obs = _env.reset()

    if not (0 <= action < _env.n_actions):
        return JSONResponse(
            {"error": f"action must be in [0, {_env.n_actions-1}], got {action}"},
            status_code=400
        )

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
        "name":        "MediTriage-Env",
        "version":     "1.0.0",
        "description": "OpenEnv medical triage environment — POST /reset to start",
        "endpoints": {
            "POST /reset": "Reset environment → returns observation",
            "POST /step":  "Take action → returns obs, reward, done, info",
            "GET  /state": "Current environment state",
            "GET  /health": "Health check",
        },
        "action_space":      {"type": "Discrete", "n": 16},
        "observation_space": {"type": "Box", "shape": [29], "dtype": "float32"},
        "tasks":             ["easy", "medium", "hard"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)