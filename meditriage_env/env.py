"""
MediTriageEnv — OpenEnv-compatible medical triage environment.

API:
    env = MediTriageEnv(task="easy")
    obs = env.reset(seed=42)
    obs, reward, done, info = env.step(action)
    state = env.state()

Action space : Discrete(16)  → (Priority × Department)
Observation  : Box(29,)       → normalised patient features
"""
from __future__ import annotations
import numpy as np
from typing import Any

from .models   import (
    OBS_DIM, NUM_ACTIONS, StepResult,
    Priority, Department,
    action_to_pd, pd_to_action,
)
from .patient_generator import generate_patient
from .reward            import compute_reward


# ── Task configurations ───────────────────────────────────────────────────────

TASK_CONFIGS: dict[str, dict] = {
    "easy": {
        "difficulty":         0,
        "n_patients":        20,
        "icu_beds":          20,
        "er_beds":           40,
        "ward_beds":         60,
        "critical_rate":     0.10,   # fraction forced-critical
        "description": (
            "20 patients, clean vitals, no resource constraints. "
            "Agent must learn basic triage classification."
        ),
    },
    "medium": {
        "difficulty":         1,
        "n_patients":        50,
        "icu_beds":           8,
        "er_beds":           20,
        "ward_beds":         30,
        "critical_rate":     0.15,
        "description": (
            "50 patients, noisy / occasionally missing vitals, "
            "moderate bed constraints. Agent must handle resource pressure."
        ),
    },
    "hard": {
        "difficulty":         2,
        "n_patients":        80,
        "icu_beds":           3,
        "er_beds":            8,
        "ward_beds":         15,
        "critical_rate":     0.25,
        "description": (
            "80 patients, high missingness, severe bed constraints, "
            "25% rare critical edge-cases. Tests robust triage under pressure."
        ),
    },
}


class MediTriageEnv:
    """
    OpenEnv-compatible Medical Triage Environment.

    Parameters
    ----------
    task : str
        One of 'easy', 'medium', 'hard'.
    """

    # OpenEnv metadata
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, task: str = "medium") -> None:
        if task not in TASK_CONFIGS:
            raise ValueError(f"task must be one of {list(TASK_CONFIGS)}, got '{task}'")

        self.task   = task
        self.cfg    = TASK_CONFIGS[task]
        self._rng   = np.random.default_rng()

        # Space descriptors (OpenEnv-compatible dicts)
        self.observation_space = {
            "type":  "Box",
            "shape": (OBS_DIM,),
            "low":   0.0,
            "high":  1.0,
            "dtype": "float32",
        }
        self.action_space = {
            "type": "Discrete",
            "n":    NUM_ACTIONS,
        }

        # Internal state (populated on reset)
        self._patients:    list = []
        self._step_idx:    int  = 0
        self._episode_rewards: list[float] = []
        self._history:     list[dict] = []
        self._done:        bool = True

        # Bed counters (dynamic)
        self._icu_beds   = 0
        self._er_beds    = 0
        self._ward_beds  = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment, return first observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = self.cfg
        self._icu_beds   = cfg["icu_beds"]
        self._er_beds    = cfg["er_beds"]
        self._ward_beds  = cfg["ward_beds"]

        n = cfg["n_patients"]
        arrival_times = np.sort(self._rng.uniform(0, 480, n))   # 8-hr shift

        self._patients = [
            generate_patient(
                patient_id   = i,
                arrival_time = float(arrival_times[i]),
                queue_length = max(0, i - self._rng.integers(0, 5)),
                difficulty   = cfg["difficulty"],
                rng          = self._rng,
                icu_beds     = self._icu_beds,
                er_beds      = self._er_beds,
                ward_beds    = self._ward_beds,
                force_critical = (self._rng.random() < cfg["critical_rate"]),
            )
            for i in range(n)
        ]

        self._step_idx         = 0
        self._episode_rewards  = []
        self._history          = []
        self._done             = False

        return self._patients[0].to_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take action for current patient.

        Returns
        -------
        observation : np.ndarray  — next patient's features (or zeros if done)
        reward      : float
        done        : bool
        info        : dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"action must be in [0, {NUM_ACTIONS-1}], got {action}")

        patient = self._patients[self._step_idx]
        reward, info = compute_reward(
            patient   = patient,
            action    = action,
            step      = self._step_idx,
            max_steps = len(self._patients),
        )

        # ── Update bed counts ──────────────────────────────────────────────────
        _, pred_dept = action_to_pd(action)
        if pred_dept == Department.ICU   and self._icu_beds   > 0:
            self._icu_beds  -= 1
        elif pred_dept == Department.ER  and self._er_beds    > 0:
            self._er_beds   -= 1
        elif pred_dept == Department.WARD and self._ward_beds > 0:
            self._ward_beds -= 1

        self._episode_rewards.append(reward)
        self._history.append({"step": self._step_idx, "action": action, **info})
        self._step_idx += 1

        done = self._step_idx >= len(self._patients)
        self._done = done

        if done:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            info["episode_return"]    = float(np.sum(self._episode_rewards))
            info["episode_mean_rew"]  = float(np.mean(self._episode_rewards))
            info["n_patients"]        = len(self._patients)
        else:
            # Update bed counts in next patient's context
            p = self._patients[self._step_idx]
            p.icu_beds_left   = self._icu_beds
            p.er_beds_left    = self._er_beds
            p.ward_beds_left  = self._ward_beds
            obs = p.to_observation()

        return obs, reward, done, info

    def state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the current environment state."""
        return {
            "task":          self.task,
            "step":          self._step_idx,
            "n_patients":    len(self._patients),
            "done":          self._done,
            "icu_beds_left": self._icu_beds,
            "er_beds_left":  self._er_beds,
            "ward_beds_left": self._ward_beds,
            "episode_return": float(np.sum(self._episode_rewards)) if self._episode_rewards else 0.0,
            "history":       self._history[-5:],   # last 5 steps only
        }

    def render(self, mode: str = "human") -> str | None:
        if self._done or self._step_idx >= len(self._patients):
            return "Episode complete."
        p = self._patients[self._step_idx]
        lines = [
            f"── Patient {p.patient_id} ({'done' if self._done else f'step {self._step_idx}/{len(self._patients)}'}) ──",
            f"  Complaint  : {p.chief_complaint.replace('_', ' ').title()}",
            f"  Age/Gender : {p.age}y / {'M' if p.gender==0 else 'F' if p.gender==1 else 'Oth'}",
            f"  HR={p.heart_rate:.0f}  BP={p.systolic_bp:.0f}/{p.diastolic_bp:.0f}  SpO2={p.spo2:.0f}%  Temp={p.temperature:.1f}°C",
            f"  Pain={p.pain_score:.1f}/10  GCS={p.gcs:.0f}",
            f"  Queue={p.queue_length}  ICU={p.icu_beds_left}  ER={p.er_beds_left}  Ward={p.ward_beds_left}",
        ]
        out = "\n".join(lines)
        if mode == "human":
            print(out)
        return out

    # ── Helpers ────────────────────────────────────────────────────────────────

    @property
    def n_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    def action_meaning(self, action: int) -> str:
        p, d = action_to_pd(action)
        return f"{p.name} → {d.name}"
