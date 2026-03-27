"""
Reward function for MediTriage-Env.

Balanced reward penalises both:
  1. Triage misclassification  (wrong priority)
  2. Routing errors             (wrong department)
  3. Critical delays            (P1/P2 sent to wrong place)
  4. Resource violations        (routing to full unit)

Partial progress: closer guesses get partial credit, not binary.
Score is normalised to [-1.0, +1.0].
"""
from __future__ import annotations
import numpy as np
from .models import Patient, Priority, Department, action_to_pd


# How many priority levels off is a given error?
_PRIORITY_DISTANCE = np.array([
    [0, 1, 2, 3],   # true=P1
    [1, 0, 1, 2],   # true=P2
    [2, 1, 0, 1],   # true=P3
    [3, 2, 1, 0],   # true=P4
], dtype=float)

# Routing compatibility: 1.0 = perfect, 0.5 = acceptable, 0.0 = wrong
# Rows=true_dept, Cols=pred_dept  [ICU, ER, WARD, DISCHARGE]
_DEPT_COMPAT = np.array([
    [1.0, 0.5, 0.0, 0.0],   # true=ICU
    [0.5, 1.0, 0.3, 0.0],   # true=ER
    [0.0, 0.3, 1.0, 0.5],   # true=WARD
    [0.0, 0.0, 0.5, 1.0],   # true=DISCHARGE
], dtype=float)

# Criticality multiplier: P1 errors are punished harder
_CRITICALITY = {
    Priority.P1_IMMEDIATE: 2.0,
    Priority.P2_EMERGENT:  1.5,
    Priority.P3_URGENT:    1.0,
    Priority.P4_NON_URGENT: 0.8,
}


def compute_reward(
    patient: Patient,
    action:  int,
    step:    int,
    max_steps: int,
) -> tuple[float, dict]:
    """
    Returns (reward, info_dict).
    reward in [-1.0, +1.0].
    """
    pred_priority, pred_dept = action_to_pd(action)
    true_p = patient.true_priority
    true_d = patient.true_department

    # ── Priority score  (1 = perfect, decreasing with distance) ──────────────
    p_dist   = _PRIORITY_DISTANCE[int(true_p), int(pred_priority)]
    p_score  = max(0.0, 1.0 - p_dist / 3.0)   # 1, 0.67, 0.33, 0

    # ── Department score ───────────────────────────────────────────────────────
    d_score  = _DEPT_COMPAT[int(true_d), int(pred_dept)]

    # ── Resource violation penalty ────────────────────────────────────────────
    resource_penalty = 0.0
    if pred_dept == Department.ICU   and patient.icu_beds_left  == 0:
        resource_penalty = 0.4
    elif pred_dept == Department.ER  and patient.er_beds_left   == 0:
        resource_penalty = 0.2
    elif pred_dept == Department.WARD and patient.ward_beds_left == 0:
        resource_penalty = 0.1

    # ── Critical patient under-triage penalty ─────────────────────────────────
    under_triage_penalty = 0.0
    if true_p == Priority.P1_IMMEDIATE and pred_priority in (
            Priority.P3_URGENT, Priority.P4_NON_URGENT):
        under_triage_penalty = 0.5
    elif true_p == Priority.P2_EMERGENT and pred_priority == Priority.P4_NON_URGENT:
        under_triage_penalty = 0.3

    # ── Weighted combined score ────────────────────────────────────────────────
    criticality = _CRITICALITY[true_p]
    raw = (
        0.45 * p_score
      + 0.45 * d_score
      - 0.05 * resource_penalty
      - 0.05 * under_triage_penalty
    ) * criticality

    # Normalise: perfect P1 = 2.0 * 0.9 = 1.8 → map to ~1.0
    reward = float(np.clip(raw / (0.9 * criticality), -1.0, 1.0))

    info = {
        "priority_score":      round(p_score, 3),
        "department_score":    round(d_score, 3),
        "resource_penalty":    round(resource_penalty, 3),
        "under_triage_penalty": round(under_triage_penalty, 3),
        "true_priority":       true_p.name,
        "true_department":     true_d.name,
        "pred_priority":       pred_priority.name,
        "pred_department":     pred_dept.name,
        "criticality":         criticality,
        "reward":              round(reward, 4),
    }
    return reward, info
