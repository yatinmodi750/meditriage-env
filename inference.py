"""
Inference Script — MediTriage-Env
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import re
import base64
import textwrap
from io import BytesIO
from typing import List, Optional, Dict

from openai import OpenAI
import numpy as np

from meditriage_env         import MediTriageEnv
from meditriage_env.schemas import PatientObservation, TriageAction
from graders.graders        import grade_all

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key-set"
MODEL_NAME   = os.getenv("MODEL_NAME")
MAX_STEPS    = 8
TEMPERATURE  = 0.2
MAX_TOKENS   = 200
FALLBACK_ACTION = "P3_URGENT,ER"

DEBUG = True

ACTION_PATTERN = re.compile(
    r'"priority"\s*:\s*"(\w+)".*?"department"\s*:\s*"(\w+)"',
    re.DOTALL,
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert emergency medicine physician performing triage.

    For each patient, decide:
    1. PRIORITY — urgency of care:
       - P1_IMMEDIATE : Life-threatening, act within minutes.
       - P2_EMERGENT  : High risk, act within 15 minutes.
       - P3_URGENT    : Stable, needs care within 1 hour.
       - P4_NON_URGENT: Minor, can safely wait 2+ hours.

    2. DEPARTMENT — where to route the patient:
       - ICU       : Intensive Care Unit (critical/life-threatening)
       - ER        : Emergency Room (urgent/emergent workup)
       - WARD      : General Ward (stable, needs admission)
       - DISCHARGE : Safe to discharge or refer outpatient

    RULES:
    - Never route to a department with 0 beds available.
    - P1/P2 patients must not go to WARD or DISCHARGE unless all options are full.
    - Missing vitals (shown as 0) may indicate monitoring failure — treat cautiously.

    Respond ONLY with valid JSON, exactly like this:
    {"priority": "P2_EMERGENT", "department": "ER"}

    No explanation. No preamble. JSON only.
    """
).strip()


def build_patient_prompt(obs: PatientObservation) -> str:
    return f"Triage this patient:\n\n{obs.to_text()}\n\nRespond with JSON only."


def parse_model_action(response_text: str) -> int:
    """Parse LLM response into an integer action. Falls back to FALLBACK_ACTION."""
    if not response_text:
        return TriageAction.from_text(FALLBACK_ACTION).to_int()

    try:
        action = TriageAction.from_text(response_text)
        return action.to_int()
    except Exception:
        if DEBUG:
            print(f"  Could not parse response: {response_text!r} — using fallback")
        return TriageAction.from_text(FALLBACK_ACTION).to_int()


def heuristic_fallback(obs_vec: np.ndarray) -> int:
    """Rule-based fallback used when LLM call fails."""
    from meditriage_env.models import Priority, Department
    from meditriage_env        import pd_to_action

    hr   = obs_vec[0] * 220;  spo2 = obs_vec[3] * 100
    sbp  = obs_vec[1] * 250;  gcs  = obs_vec[6] * 15
    pain = obs_vec[5] * 10
    icu  = obs_vec[26] * 20 > 0
    er   = obs_vec[27] * 40 > 0

    if gcs < 9 or spo2 < 88 or sbp < 75 or hr > 160:
        prio = Priority.P1_IMMEDIATE
    elif gcs < 13 or spo2 < 93 or sbp < 95 or pain >= 8:
        prio = Priority.P2_EMERGENT
    elif pain >= 5 or spo2 < 96:
        prio = Priority.P3_URGENT
    else:
        prio = Priority.P4_NON_URGENT

    if prio == Priority.P1_IMMEDIATE and icu:
        dept = Department.ICU
    elif prio in (Priority.P1_IMMEDIATE, Priority.P2_EMERGENT) and er:
        dept = Department.ER
    elif prio == Priority.P3_URGENT:
        dept = Department.WARD
    else:
        dept = Department.DISCHARGE

    return pd_to_action(prio, dept)


def main() -> None:
    import json, time

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 62)
    print("  MediTriage-Env — Inference Script")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  API_KEY set  : {'yes' if API_KEY else 'NO — set HF_TOKEN or API_KEY'}")
    print("=" * 62)

    if not API_KEY:
        print("\n❌  No API key. Set HF_TOKEN or API_KEY.")
        return

    if not MODEL_NAME:
        print("\n❌  MODEL_NAME not set.")
        return

    def agent(obs_vec: np.ndarray, state: dict) -> int:
        step       = state["step"]
        n_patients = state["n_patients"]

        obs = PatientObservation.from_vector(
            list(obs_vec.astype(float)),
            patient_id  = step,
            step        = step,
            total_steps = n_patients,
        )

        user_prompt = build_patient_prompt(obs)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        try:
            completion = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                stream      = False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            failure_msg = f"Model request failed ({exc}). Using fallback action."
            print(failure_msg)
            return heuristic_fallback(obs_vec)

        action_int = parse_model_action(response_text)

        if DEBUG:
            action_obj = TriageAction.from_int(action_int)
            print(f"  Step {step+1:02d}/{n_patients} → {action_obj.priority} / {action_obj.department}")

        return action_int

    N_EPS = 5
    print(f"\nRunning {N_EPS} episodes per task...\n")

    t0      = time.time()
    results = grade_all(agent, n_episodes=N_EPS)
    elapsed = time.time() - t0

    print("\n" + "=" * 62)
    print("  Results")
    print("=" * 62)
    print(f"  Overall score : {results['overall_score']:.4f}")
    for task in ("easy", "medium", "hard"):
        print(f"  {task.capitalize():7s} score : {results[task]['score']:.4f}")
    print(f"\n  Elapsed : {elapsed:.1f}s  |  Model : {MODEL_NAME}")

    out = {
        "model":           MODEL_NAME,
        "api_base_url":    API_BASE_URL,
        "n_episodes":      N_EPS,
        "results":         results,
        "elapsed_seconds": round(elapsed, 2),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Saved → baseline_results.json")
    print("\nDone. ✓")


if __name__ == "__main__":
    main()