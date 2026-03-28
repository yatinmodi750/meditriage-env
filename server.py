"""
server.py — FastAPI HTTP server exposing the OpenEnv API.

Endpoints:
    POST /reset          → {"seed": int | null}         → observation + state
    POST /step           → {"action": int}               → obs, reward, done, info
    GET  /state          → {}                            → current state
    GET  /health         → {}                            → health check
    POST /reset/{task}   → reset with specific task
"""
from __future__ import annotations
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import uvicorn

from meditriage_env import MediTriageEnv
from meditriage_env.schemas import PatientObservation, TriageReward, EnvState

app = FastAPI(
    title       = "MediTriage-Env API",
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

# ── Global environment instance ───────────────────────────────────────────────
_env: MediTriageEnv = MediTriageEnv(task="medium")
_last_obs: Optional[np.ndarray] = None


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task: Optional[str] = "medium"

class StepRequest(BaseModel):
    action: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def obs_to_list(obs: np.ndarray) -> list:
    return [round(float(x), 6) for x in obs]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "MediTriage-Env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env, _last_obs

    task = req.task or "medium"
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"task must be easy/medium/hard, got '{task}'")

    _env      = MediTriageEnv(task=task)
    _last_obs = _env.reset(seed=req.seed)
    state     = _env.state()

    return {
        "observation": obs_to_list(_last_obs),
        "state":       state,
        "task":        task,
        "obs_dim":     len(_last_obs),
        "n_actions":   _env.n_actions,
    }


@app.post("/step")
def step(req: StepRequest):
    global _last_obs

    if _env._done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")
    if not (0 <= req.action < _env.n_actions):
        raise HTTPException(status_code=400, detail=f"action must be in [0, {_env.n_actions-1}]")

    obs, reward, done, info = _env.step(req.action)
    _last_obs = obs

    return {
        "observation": obs_to_list(obs),
        "reward":      round(float(reward), 6),
        "done":        done,
        "info":        info,
        "state":       _env.state(),
    }


@app.get("/state")
def state():
    return _env.state()


@app.get("/")
def root():
    return {
        "name":        "MediTriage-Env",
        "version":     "1.0.0",
        "description": "OpenEnv medical triage environment",
        "endpoints": {
            "POST /reset": "Reset environment, returns initial observation",
            "POST /step":  "Take action, returns obs/reward/done/info",
            "GET /state":  "Get current environment state",
            "GET /health": "Health check",
        },
        "action_space":      {"type": "Discrete", "n": 16},
        "observation_space": {"type": "Box", "shape": [29]},
        "tasks":             ["easy", "medium", "hard"],
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)