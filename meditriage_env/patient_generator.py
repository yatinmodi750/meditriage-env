"""
Procedural patient generator.
Produces statistically plausible patients whose vitals correlate with
ground-truth priority / department.  Difficulty level controls:
  - noise on vitals
  - proportion of missing vitals (mask)
  - rare critical edge cases
  - resource constraints (bed limits)
"""
from __future__ import annotations
import numpy as np
from .models import (
    Patient, Priority, Department,
    CHIEF_COMPLAINTS,
)

RNG = np.random.default_rng()   # module-level, re-seeded by env


# ── Ground-truth mapping ───────────────────────────────────────────────────────

# (complaint) → (priority_distribution, most_likely_dept)
_COMPLAINT_PROFILE: dict[str, tuple[list[float], Department]] = {
    "chest_pain":         ([0.35, 0.40, 0.20, 0.05], Department.ER),
    "dyspnea":            ([0.25, 0.40, 0.30, 0.05], Department.ER),
    "altered_consciousness": ([0.50, 0.30, 0.15, 0.05], Department.ICU),
    "trauma":             ([0.30, 0.35, 0.25, 0.10], Department.ER),
    "sepsis_signs":       ([0.45, 0.35, 0.15, 0.05], Department.ICU),
    "stroke_symptoms":    ([0.50, 0.35, 0.10, 0.05], Department.ICU),
    "abdominal_pain":     ([0.10, 0.25, 0.45, 0.20], Department.ER),
    "fracture":           ([0.05, 0.20, 0.55, 0.20], Department.WARD),
    "laceration":         ([0.05, 0.15, 0.45, 0.35], Department.ER),
    "fever_infection":    ([0.05, 0.20, 0.50, 0.25], Department.WARD),
    "syncope":            ([0.20, 0.35, 0.35, 0.10], Department.ER),
    "allergic_reaction":  ([0.30, 0.35, 0.25, 0.10], Department.ER),
    "hypertensive_crisis":([0.30, 0.40, 0.25, 0.05], Department.ER),
    "diabetic_emergency": ([0.25, 0.35, 0.30, 0.10], Department.ER),
    "psychiatric_crisis": ([0.10, 0.20, 0.40, 0.30], Department.WARD),
}

# Vitals ranges by priority: (mean, std) for [HR, SBP, DBP, SpO2, Temp, Pain, GCS]
_VITALS_BY_PRIORITY: dict[Priority, tuple[list, list]] = {
    Priority.P1_IMMEDIATE: (
        [120, 80,  50, 88, 38.8, 8.5, 8],
        [25,  25,  15, 5,  1.0,  1.5, 3],
    ),
    Priority.P2_EMERGENT: (
        [105, 100, 65, 93, 38.3, 7.0, 12],
        [20,  20,  12, 4,  0.8,  1.5, 2],
    ),
    Priority.P3_URGENT: (
        [90,  125, 80, 96, 37.8, 5.0, 14],
        [15,  18,  10, 3,  0.7,  2.0, 1],
    ),
    Priority.P4_NON_URGENT: (
        [78,  125, 78, 98, 37.2, 2.5, 15],
        [12,  15,  10, 2,  0.5,  1.5, 0],
    ),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def generate_patient(
    patient_id:    int,
    arrival_time:  float,
    queue_length:  int,
    difficulty:    int,          # 0=easy, 1=medium, 2=hard
    rng:           np.random.Generator,
    icu_beds:      int = 99,
    er_beds:       int = 99,
    ward_beds:     int = 99,
    force_critical: bool = False,
) -> Patient:
    """Sample a single patient."""

    # ── Complaint ──────────────────────────────────────────────────────────────
    if force_critical:
        critical = ["altered_consciousness", "sepsis_signs", "stroke_symptoms",
                    "chest_pain", "dyspnea"]
        complaint = rng.choice(critical)
    else:
        complaint = rng.choice(CHIEF_COMPLAINTS)

    profile, dept = _COMPLAINT_PROFILE[complaint]

    # ── True priority ──────────────────────────────────────────────────────────
    priority = Priority(rng.choice(4, p=profile))

    # ── True department (resource pressure may force re-route) ─────────────────
    department = dept
    if difficulty >= 1:
        if dept == Department.ICU  and icu_beds  == 0:
            department = Department.ER
        elif dept == Department.ER   and er_beds   == 0:
            department = Department.WARD
        elif dept == Department.WARD and ward_beds == 0:
            department = Department.DISCHARGE

    # ── Vitals ─────────────────────────────────────────────────────────────────
    means, stds = _VITALS_BY_PRIORITY[priority]
    noise_scale = 1.0 + difficulty * 0.5        # more noise at harder levels
    vitals_raw = rng.normal(means, [s * noise_scale for s in stds])

    hr   = _clamp(vitals_raw[0], 30,  220)
    sbp  = _clamp(vitals_raw[1], 60,  250)
    dbp  = _clamp(vitals_raw[2], 30,  150)
    spo2 = _clamp(vitals_raw[3], 70,  100)
    temp = _clamp(vitals_raw[4], 35,  42)
    pain = _clamp(vitals_raw[5],  0,  10)
    gcs  = _clamp(round(vitals_raw[6]), 3, 15)

    # ── Missing vitals mask (harder → more missing) ────────────────────────────
    missing_prob = [0.0, 0.10, 0.25][difficulty]
    mask = (rng.random(7) > missing_prob).astype(float)
    mask[0] = 1.0   # HR always present

    # ── Demographics ───────────────────────────────────────────────────────────
    age    = int(_clamp(rng.normal(52, 20), 1, 100))
    gender = int(rng.choice(3, p=[0.49, 0.49, 0.02]))

    return Patient(
        patient_id      = patient_id,
        age             = age,
        gender          = gender,
        chief_complaint = complaint,
        heart_rate      = hr,
        systolic_bp     = sbp,
        diastolic_bp    = dbp,
        spo2            = spo2,
        temperature     = temp,
        pain_score      = pain,
        gcs             = gcs,
        arrival_time    = arrival_time,
        queue_length    = queue_length,
        true_priority   = priority,
        true_department = department,
        icu_beds_left   = icu_beds,
        er_beds_left    = er_beds,
        ward_beds_left  = ward_beds,
        vitals_mask     = mask,
    )
