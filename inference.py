"""
Inference Script — MediTriage-Env
===================================
MANDATORY
- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI
import numpy as np

from meditriage_env         import MediTriageEnv
from meditriage_env.schemas import PatientObservation, TriageAction

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK             = "meditriage-env"
MAX_STEPS             = 80
TEMPERATURE           = 0.2
MAX_TOKENS            = 200
SUCCESS_SCORE_THRESHOLD = 0.3

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert emergency medicine physician performing triage.
    For each patient, decide priority and department.
    Priority: P1_IMMEDIATE, P2_EMERGENT, P3_URGENT, P4_NON_URGENT
    Department: ICU, ER, WARD, DISCHARGE
    Respond ONLY with valid JSON: {"priority": "P2_EMERGENT", "department": "ER"}
""").strip()


# ── Structured log helpers ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Agent helpers ─────────────────────────────────────────────────────────────

def heuristic_fallback(obs_vec: np.ndarray) -> int:
    from meditriage_env.models import Priority, Department
    from meditriage_env        import pd_to_action
    hr=obs_vec[0]*220; spo2=obs_vec[3]*100; sbp=obs_vec[1]*250
    gcs=obs_vec[6]*15; pain=obs_vec[5]*10
    icu=obs_vec[26]*20>0; er=obs_vec[27]*40>0
    if gcs<9 or spo2<88 or sbp<75 or hr>160: prio=Priority.P1_IMMEDIATE
    elif gcs<13 or spo2<93 or sbp<95 or pain>=8: prio=Priority.P2_EMERGENT
    elif pain>=5 or spo2<96: prio=Priority.P3_URGENT
    else: prio=Priority.P4_NON_URGENT
    if prio==Priority.P1_IMMEDIATE and icu: dept=Department.ICU
    elif prio in(Priority.P1_IMMEDIATE,Priority.P2_EMERGENT) and er: dept=Department.ER
    elif prio==Priority.P3_URGENT: dept=Department.WARD
    else: dept=Department.DISCHARGE
    return pd_to_action(prio, dept)


def get_action(client: OpenAI, obs_vec: np.ndarray, state: dict) -> tuple:
    step = state["step"]; n = state["n_patients"]
    obs  = PatientObservation.from_vector(list(obs_vec.astype(float)), patient_id=step, step=step, total_steps=n)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Triage this patient:\n\n{obs.to_text()}\n\nJSON only."},
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=False,
        )
        text   = (completion.choices[0].message.content or "").strip()
        action = TriageAction.from_text(text)
        return action.to_int(), f"{action.priority}/{action.department}"
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        action_int = heuristic_fallback(obs_vec)
        action_obj = TriageAction.from_int(action_int)
        return action_int, f"{action_obj.priority}/{action_obj.department}"


def run_task(client: OpenAI, task_id: str) -> None:
    """Run one full episode for a task and emit START/STEP/END logs."""
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.01
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = MediTriageEnv(task=task_id)
        obs = env.reset(seed=42)

        for step in range(1, MAX_STEPS + 1):
            if env._done:
                break

            action_int, action_str = get_action(client, obs, env.state())
            obs, reward, done, info = env.step(action_int)

            reward = float(reward)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Normalise mean reward [-1,1] → (0.01, 0.99)
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        raw_score   = (mean_reward + 1.0) / 2.0
        score       = max(0.01, min(0.99, raw_score))
        success     = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        score   = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key-set")

    # Run ALL 3 tasks — checker expects 3 START/END pairs
    for task_id in ["easy", "medium", "hard"]:
        run_task(client, task_id)


if __name__ == "__main__":
    main()