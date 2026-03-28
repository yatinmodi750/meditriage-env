"""
baseline_inference.py — LLM-powered baseline for MediTriage-Env.

Uses the OpenAI API client to run an LLM agent against all 3 tasks.
Reads credentials from environment variables:
    OPENAI_API_KEY   (required)
    OPENAI_BASE_URL  (optional, defaults to https://api.openai.com/v1)
    OPENAI_MODEL     (optional, defaults to gpt-4o-mini)

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/baseline_inference.py

    # Or with a custom model / base URL (e.g. for OpenRouter):
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    OPENAI_MODEL=mistralai/mistral-7b-instruct \
    python scripts/baseline_inference.py
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from openai import OpenAI

from meditriage_env          import MediTriageEnv, NUM_ACTIONS
from meditriage_env.schemas  import PatientObservation, TriageAction
from graders.graders         import grade_all


# ── OpenAI client setup ───────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key  = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Run: export OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are an expert emergency medicine physician performing triage.

For each patient, you must decide:
1. PRIORITY — how urgently they need care:
   - P1_IMMEDIATE : Life-threatening. Act within minutes.
   - P2_EMERGENT  : High risk. Act within 15 minutes.
   - P3_URGENT    : Stable but needs care within 1 hour.
   - P4_NON_URGENT: Minor complaint, can safely wait 2+ hours.

2. DEPARTMENT — where to route them:
   - ICU       : Intensive Care Unit (for critical, life-threatening cases)
   - ER        : Emergency Room (for urgent/emergent cases needing immediate workup)
   - WARD      : General Ward (for stable patients needing admission)
   - DISCHARGE : Safe to discharge or refer to outpatient care

CRITICAL RULES:
- Never route a patient to a department with 0 beds available.
- P1/P2 patients must never be sent to WARD or DISCHARGE unless all other options are full.
- Consider ALL vitals carefully — missing values (shown as 0) may indicate monitoring failure.

Respond ONLY with valid JSON in this exact format:
{"priority": "P2_EMERGENT", "department": "ER"}

No explanation. No preamble. JSON only."""


def make_user_prompt(obs: PatientObservation) -> str:
    return f"Triage this patient:\n\n{obs.to_text()}\n\nRespond with JSON only."


# ── LLM Agent ─────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Calls OpenAI API for each triage decision.
    Falls back to heuristic if API call fails.
    """

    def __init__(self, client: OpenAI, model: str, verbose: bool = False):
        self.client  = client
        self.model   = model
        self.verbose = verbose

    def __call__(self, obs_vec: np.ndarray, state: dict) -> int:
        step       = state["step"]
        n_patients = state["n_patients"]

        obs = PatientObservation.from_vector(
            list(obs_vec.astype(float)),
            patient_id  = step,
            step        = step,
            total_steps = n_patients,
        )

        try:
            response = self.client.chat.completions.create(
                model    = self.model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": make_user_prompt(obs)},
                ],
                temperature = 0.0,
                max_tokens  = 60,
            )
            raw_text = response.choices[0].message.content.strip()
            action   = TriageAction.from_text(raw_text)

            if self.verbose:
                print(f"  Step {step+1:02d}/{n_patients} | {action.priority} → {action.department}")

            return action.to_int()

        except Exception as e:
            if self.verbose:
                print(f"  Step {step+1:02d} | API error ({e}) — heuristic fallback")
            return self._heuristic_fallback(obs_vec)

    def _heuristic_fallback(self, obs: np.ndarray) -> int:
        from meditriage_env.models import Priority, Department
        from meditriage_env        import pd_to_action

        hr = obs[0] * 220; spo2 = obs[3] * 100
        sbp = obs[1] * 250; gcs = obs[6] * 15; pain = obs[5] * 10
        icu = obs[26] * 20 > 0; er = obs[27] * 40 > 0

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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  MediTriage-Env — LLM Baseline Inference")
    print(f"  Model : {MODEL}")
    print("=" * 62)

    try:
        client = get_client()
    except EnvironmentError as e:
        print(f"\n❌  {e}")
        sys.exit(1)

    agent   = LLMAgent(client, MODEL, verbose=True)
    N_EPS   = 5   # keep low to manage API costs

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
    print(f"\n  Time elapsed  : {elapsed:.1f}s  |  Model: {MODEL}")

    # Save results for reproducibility
    out_path = os.path.join(os.path.dirname(__file__), "..", "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":           MODEL,
            "n_episodes":      N_EPS,
            "results":         results,
            "elapsed_seconds": round(elapsed, 2),
        }, f, indent=2)
    print(f"  Results saved → baseline_results.json")
    print("\nDone. ✓")