"""
baseline_inference.py — reproducible baseline scores for MediTriage-Env.

Agents implemented:
  1. RandomAgent         — uniform random action
  2. HeuristicAgent      — rule-based vitals thresholds
  3. OptimalOracleAgent  — cheats with ground truth (upper bound)

Run:
    python scripts/baseline_inference.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from meditriage_env import MediTriageEnv, NUM_ACTIONS, action_to_pd, pd_to_action
from meditriage_env.models import Priority, Department
from graders.graders import grade_all


# ── 1. Random Agent ───────────────────────────────────────────────────────────

_rng = np.random.default_rng(0)

def random_agent(obs: np.ndarray, state: dict) -> int:
    return int(_rng.integers(0, NUM_ACTIONS))


# ── 2. Heuristic Agent ────────────────────────────────────────────────────────

def heuristic_agent(obs: np.ndarray, state: dict) -> int:
    """
    Rule-based agent using normalised observation indices.
    Obs layout: [HR, SBP, DBP, SpO2, Temp, Pain, GCS,  ← indices 0-6
                 Age, Gender,                           ← 7-8
                 complaint_one_hot (15),                ← 9-23
                 arrival_time, queue, icu, er, ward]    ← 24-28
    """
    hr   = obs[0] * 220
    sbp  = obs[1] * 250
    spo2 = obs[3] * 100
    gcs  = obs[6] * 15
    pain = obs[5] * 10

    # ── Priority heuristic ────────────────────────────────────────────────────
    if gcs < 9 or spo2 < 88 or sbp < 75 or hr > 160:
        priority = Priority.P1_IMMEDIATE
    elif gcs < 13 or spo2 < 93 or sbp < 95 or pain >= 8:
        priority = Priority.P2_EMERGENT
    elif pain >= 5 or spo2 < 96:
        priority = Priority.P3_URGENT
    else:
        priority = Priority.P4_NON_URGENT

    # ── Department heuristic ──────────────────────────────────────────────────
    icu_avail  = obs[26] * 20 > 0   # icu_beds_left
    er_avail   = obs[27] * 40 > 0

    if priority == Priority.P1_IMMEDIATE and icu_avail:
        dept = Department.ICU
    elif priority in (Priority.P1_IMMEDIATE, Priority.P2_EMERGENT) and er_avail:
        dept = Department.ER
    elif priority == Priority.P3_URGENT:
        dept = Department.WARD
    else:
        dept = Department.DISCHARGE

    return pd_to_action(priority, dept)


# ── 3. Oracle Agent (upper bound) ─────────────────────────────────────────────

class OracleAgent:
    """Peeks at ground truth — not a valid agent, sets upper bound only."""
    def __init__(self, task: str):
        self._env = MediTriageEnv(task=task)

    def __call__(self, obs: np.ndarray, state: dict) -> int:
        step = state["step"]
        if step >= len(self._env._patients):
            return 0
        p = self._env._patients[step]
        return pd_to_action(p.true_priority, p.true_department)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N = 20   # episodes per grader

    print("=" * 60)
    print("  MediTriage-Env  —  Baseline Inference Report")
    print("=" * 60)

    for name, agent in [("Random", random_agent), ("Heuristic", heuristic_agent)]:
        print(f"\n▶  Agent: {name}")
        results = grade_all(agent, n_episodes=N)
        print(f"   Overall score : {results['overall_score']:.4f}")
        for task in ("easy", "medium", "hard"):
            t = results[task]
            print(f"   {task.capitalize():7s} score : {t['score']:.4f}   ({t['description'][:55]})")

    print("\n" + "=" * 60)
    print("  Oracle upper bounds (cheats with ground truth)")
    print("=" * 60)
    for task in ("easy", "medium", "hard"):
        oracle = OracleAgent(task)
        # Pre-populate patients by resetting
        env = MediTriageEnv(task=task)
        oracle._env = env
        scores = []
        for ep in range(N):
            obs = env.reset(seed=ep)
            oracle._env = env
            done = False
            ep_rewards = []
            while not done:
                action = oracle(obs, env.state())
                obs, r, done, _ = env.step(action)
                ep_rewards.append(r)
            scores.append(np.mean(ep_rewards))
        print(f"   {task.capitalize():7s} mean reward: {np.mean(scores):.4f}")

    print("\nDone. ✓")
