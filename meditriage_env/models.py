"""
MediTriage-Env: Typed models for patient state, actions, and observations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np


# ── Action Spaces ──────────────────────────────────────────────────────────────

class Priority(IntEnum):
    """ESI-style triage priority levels."""
    P1_IMMEDIATE   = 0   # Life-threatening, act within minutes
    P2_EMERGENT    = 1   # High risk, act within 15 min
    P3_URGENT      = 2   # Stable but needs care within 1 hr
    P4_NON_URGENT  = 3   # Minor, can wait 2+ hrs


class Department(IntEnum):
    """Available routing destinations."""
    ICU       = 0   # Intensive care unit
    ER        = 1   # Emergency room
    WARD      = 2   # General ward
    DISCHARGE = 3   # Safe to discharge / refer out


# Combined action: (priority, department) — 4×4 = 16 discrete actions
NUM_ACTIONS = len(Priority) * len(Department)   # 16


def action_to_pd(action: int) -> tuple[Priority, Department]:
    return Priority(action // len(Department)), Department(action % len(Department))


def pd_to_action(p: Priority, d: Department) -> int:
    return int(p) * len(Department) + int(d)


# ── Patient ────────────────────────────────────────────────────────────────────

CHIEF_COMPLAINTS = [
    "chest_pain", "dyspnea", "altered_consciousness", "trauma",
    "sepsis_signs", "stroke_symptoms", "abdominal_pain",
    "fracture", "laceration", "fever_infection",
    "syncope", "allergic_reaction", "hypertensive_crisis",
    "diabetic_emergency", "psychiatric_crisis",
]

COMPLAINT_TO_IDX = {c: i for i, c in enumerate(CHIEF_COMPLAINTS)}


@dataclass
class Patient:
    patient_id:      int
    age:             int          # years
    gender:          int          # 0=M, 1=F, 2=Other
    chief_complaint: str

    # Vitals
    heart_rate:      float        # bpm
    systolic_bp:     float        # mmHg
    diastolic_bp:    float        # mmHg
    spo2:            float        # % oxygen saturation
    temperature:     float        # °C

    # Clinical scores
    pain_score:      float        # 0-10 NRS
    gcs:             float        # Glasgow Coma Scale 3-15

    # Queue context
    arrival_time:    float        # minutes since episode start
    queue_length:    int          # patients waiting at arrival

    # Ground truth (hidden from agent, used by grader)
    true_priority:   Priority     = Priority.P3_URGENT
    true_department: Department   = Department.ER

    # Resource flags (set by environment for harder tasks)
    icu_beds_left:   int          = 99
    er_beds_left:    int          = 99
    ward_beds_left:  int          = 99

    # Missing-data mask (1 = observed, 0 = missing)
    vitals_mask:     np.ndarray   = field(default_factory=lambda: np.ones(7))

    def to_observation(self) -> np.ndarray:
        """Return a flat float32 observation vector (length = OBS_DIM)."""
        complaint_vec = np.zeros(len(CHIEF_COMPLAINTS), dtype=np.float32)
        complaint_vec[COMPLAINT_TO_IDX.get(self.chief_complaint, 0)] = 1.0

        vitals = np.array([
            self.heart_rate    / 220.0,
            self.systolic_bp   / 250.0,
            self.diastolic_bp  / 150.0,
            self.spo2          / 100.0,
            self.temperature   / 42.0,
            self.pain_score    / 10.0,
            self.gcs           / 15.0,
        ], dtype=np.float32) * self.vitals_mask.astype(np.float32)

        demographics = np.array([
            self.age / 100.0,
            self.gender / 2.0,
        ], dtype=np.float32)

        context = np.array([
            self.arrival_time / 480.0,       # normalise to 8-hr shift
            self.queue_length / 50.0,
            self.icu_beds_left  / 20.0,
            self.er_beds_left   / 40.0,
            self.ward_beds_left / 60.0,
        ], dtype=np.float32)

        return np.concatenate([vitals, demographics, complaint_vec, context])


# Observation space dimension
OBS_DIM = 7 + 2 + len(CHIEF_COMPLAINTS) + 5   # = 29


@dataclass
class StepResult:
    observation:  np.ndarray
    reward:       float
    done:         bool
    info:         dict
