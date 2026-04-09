"""
server/app.py — OpenEnv-compliant FastAPI server for MediTriage-Env.
Exposes: /reset, /step, /state, /tasks, /grader, /health
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def strict_score(v: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return max(0.01, min(0.99, float(v)))


# ── Task catalog ──────────────────────────────────────────────────────────────

TASKS = [
    {
        "id":          "easy",
        "name":        "Easy Triage",
        "description": "20 patients, clean vitals, ample resources. Learn basic triage.",
        "difficulty":  "easy",
        "n_patients":  20,
    },
    {
        "id":          "medium",
        "name":        "Medium Triage",
        "description": "50 patients, noisy/missing vitals, moderate bed constraints.",
        "difficulty":  "medium",
        "n_patients":  50,
    },
    {
        "id":          "hard",
        "name":        "Hard Triage",
        "description": "80 patients, severe constraints, 25% critical edge cases.",
        "difficulty":  "hard",
        "n_patients":  80,
    },
]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "env": "MediTriage-Env", "version": "1.0.0"}


@app.get("/tasks")
async def get_tasks():
    """Return the task catalog — required by openenv validate."""
    return JSONResponse({"tasks": TASKS})


@app.post("/grader")
async def grader(request: Request):
    """
    Grade an agent on a specific task.
    Body: {"task_id": "easy"|"medium"|"hard", "n_episodes": 1}
    Returns: {"task_id": ..., "score": float strictly in (0,1)}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id    = body.get("task_id", "medium")
    n_episodes = int(body.get("n_episodes", 1))

    if task_id not in ("easy", "medium", "hard"):
        return JSONResponse({"error": f"Unknown task_id: {task_id}"}, status_code=400)

    # Run a random agent for grading baseline
    import numpy as _np
    env = MediTriageEnv(task=task_id)
    all_rewards = []

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        done = False
        while not done:
            action = int(_np.random.randint(0, env.n_actions))
            obs, reward, done, _ = env.step(action)
            all_rewards.append(float(reward))

    mean_reward = float(_np.mean(all_rewards)) if all_rewards else 0.0
    raw_score   = (mean_reward + 1.0) / 2.0
    score       = strict_score(raw_score)

    return JSONResponse({
        "task_id":    task_id,
        "score":      score,
        "n_episodes": n_episodes,
        "mean_reward": round(mean_reward, 4),
    })


@app.post("/reset")
async def reset(request: Request):
    global _env, _last_obs
    try:
        body = await request.json()
    except Exception:
        body = {}

    seed = body.get("seed", None)
    task = body.get("task", body.get("task_id", "medium"))
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
        "tasks":             [t["id"] for t in TASKS],
        "endpoints": {
            "GET  /tasks":   "Task catalog",
            "POST /grader":  "Grade agent on a task",
            "POST /reset":   "Reset environment",
            "POST /step":    "Take action",
            "GET  /state":   "Current state",
            "GET  /health":  "Health check",
        }
    })


def main():
    """Entry point for openenv validate."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()