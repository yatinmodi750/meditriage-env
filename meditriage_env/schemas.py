"""
OpenEnv-compliant Pydantic schemas for MediTriage-Env.

Typed models for:
  - PatientObservation  (what the agent sees)
  - TriageAction        (what the agent does)
  - TriageReward        (what reward signal looks like)
  - EnvState            (state() return type)
  - StepResponse        (step() return type)
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Observation ───────────────────────────────────────────────────────────────

class PatientObservation(BaseModel):
    """
    Full observation for one incoming patient.
    All continuous values are normalised to [0, 1].
    Missing vitals are represented as 0.0.
    """

    # Vitals (normalised)
    heart_rate:       float = Field(..., ge=0.0, le=1.0, description="Heart rate / 220")
    systolic_bp:      float = Field(..., ge=0.0, le=1.0, description="Systolic BP / 250")
    diastolic_bp:     float = Field(..., ge=0.0, le=1.0, description="Diastolic BP / 150")
    spo2:             float = Field(..., ge=0.0, le=1.0, description="SpO2 / 100")
    temperature:      float = Field(..., ge=0.0, le=1.0, description="Temperature / 42")
    pain_score:       float = Field(..., ge=0.0, le=1.0, description="Pain score / 10")
    gcs:              float = Field(..., ge=0.0, le=1.0, description="GCS / 15")

    # Demographics (normalised)
    age:              float = Field(..., ge=0.0, le=1.0, description="Age / 100")
    gender:           float = Field(..., ge=0.0, le=1.0, description="0=M, 0.5=F, 1=Other")

    # Chief complaint (one-hot, 15 values)
    chief_complaint_vec: List[float] = Field(
        ..., min_length=15, max_length=15,
        description="One-hot encoding of chief complaint (15 categories)"
    )

    # Queue / resource context (normalised)
    arrival_time:     float = Field(..., ge=0.0, le=1.0, description="Arrival time / 480 min")
    queue_length:     float = Field(..., ge=0.0, le=1.0, description="Queue length / 50")
    icu_beds:         float = Field(..., ge=0.0, le=1.0, description="ICU beds remaining / 20")
    er_beds:          float = Field(..., ge=0.0, le=1.0, description="ER beds remaining / 40")
    ward_beds:        float = Field(..., ge=0.0, le=1.0, description="Ward beds remaining / 60")

    # Metadata
    patient_id:       int   = Field(..., description="Patient index within episode")
    step:             int   = Field(..., description="Current step number")
    total_steps:      int   = Field(..., description="Total patients in episode")

    @field_validator("chief_complaint_vec")
    @classmethod
    def must_be_one_hot(cls, v: List[float]) -> List[float]:
        s = sum(v)
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"chief_complaint_vec must sum to 1.0, got {s:.3f}")
        return v

    def to_vector(self) -> List[float]:
        """Return flat 29-dim list matching env observation space."""
        return [
            self.heart_rate, self.systolic_bp, self.diastolic_bp,
            self.spo2, self.temperature, self.pain_score, self.gcs,
            self.age, self.gender,
            *self.chief_complaint_vec,
            self.arrival_time, self.queue_length,
            self.icu_beds, self.er_beds, self.ward_beds,
        ]

    @classmethod
    def from_vector(cls, vec: List[float], patient_id: int, step: int, total_steps: int) -> "PatientObservation":
        """Reconstruct from a flat 29-dim observation vector."""
        if len(vec) != 29:
            raise ValueError(f"Expected 29-dim vector, got {len(vec)}")
        return cls(
            heart_rate           = vec[0],
            systolic_bp          = vec[1],
            diastolic_bp         = vec[2],
            spo2                 = vec[3],
            temperature          = vec[4],
            pain_score           = vec[5],
            gcs                  = vec[6],
            age                  = vec[7],
            gender               = vec[8],
            chief_complaint_vec  = vec[9:24],
            arrival_time         = vec[24],
            queue_length         = vec[25],
            icu_beds             = vec[26],
            er_beds              = vec[27],
            ward_beds            = vec[28],
            patient_id           = patient_id,
            step                 = step,
            total_steps          = total_steps,
        )

    def to_text(self) -> str:
        """Human-readable summary for LLM agent prompts."""
        from .models import CHIEF_COMPLAINTS
        complaint_idx = self.chief_complaint_vec.index(max(self.chief_complaint_vec))
        complaint     = CHIEF_COMPLAINTS[complaint_idx].replace("_", " ").title()
        gender_str    = {0.0: "Male", 0.5: "Female"}.get(round(self.gender * 2) / 2, "Other")

        return f"""Patient {self.patient_id + 1} of {self.total_steps}
Chief Complaint : {complaint}
Age / Gender    : {int(self.age * 100)}y / {gender_str}
Heart Rate      : {self.heart_rate * 220:.0f} bpm
Blood Pressure  : {self.systolic_bp * 250:.0f}/{self.diastolic_bp * 150:.0f} mmHg
SpO2            : {self.spo2 * 100:.0f}%
Temperature     : {self.temperature * 42:.1f} °C
Pain Score      : {self.pain_score * 10:.1f} / 10
GCS             : {self.gcs * 15:.0f} / 15
Arrival Time    : {self.arrival_time * 480:.0f} min into shift
Queue Length    : {int(self.queue_length * 50)} patients waiting
Beds Available  : ICU={int(self.icu_beds * 20)}  ER={int(self.er_beds * 40)}  Ward={int(self.ward_beds * 60)}"""


# ── Action ────────────────────────────────────────────────────────────────────

VALID_PRIORITIES   = ["P1_IMMEDIATE", "P2_EMERGENT", "P3_URGENT", "P4_NON_URGENT"]
VALID_DEPARTMENTS  = ["ICU", "ER", "WARD", "DISCHARGE"]


class TriageAction(BaseModel):
    """
    The agent's triage decision for one patient.
    Can be constructed from a priority+department pair or a raw integer action.
    """
    priority:   str = Field(..., description="One of: P1_IMMEDIATE, P2_EMERGENT, P3_URGENT, P4_NON_URGENT")
    department: str = Field(..., description="One of: ICU, ER, WARD, DISCHARGE")

    @field_validator("priority")
    @classmethod
    def valid_priority(cls, v: str) -> str:
        if v not in VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {VALID_PRIORITIES}, got '{v}'")
        return v

    @field_validator("department")
    @classmethod
    def valid_department(cls, v: str) -> str:
        if v not in VALID_DEPARTMENTS:
            raise ValueError(f"department must be one of {VALID_DEPARTMENTS}, got '{v}'")
        return v

    def to_int(self) -> int:
        """Convert to integer action in [0, 15]."""
        p = VALID_PRIORITIES.index(self.priority)
        d = VALID_DEPARTMENTS.index(self.department)
        return p * 4 + d

    @classmethod
    def from_int(cls, action: int) -> "TriageAction":
        """Reconstruct from integer action."""
        if not (0 <= action <= 15):
            raise ValueError(f"action must be in [0, 15], got {action}")
        return cls(
            priority   = VALID_PRIORITIES[action // 4],
            department = VALID_DEPARTMENTS[action % 4],
        )

    @classmethod
    def from_text(cls, text: str) -> "TriageAction":
        """
        Parse LLM output into a TriageAction.
        Expects JSON like: {"priority": "P2_EMERGENT", "department": "ER"}
        Falls back to keyword search if JSON parsing fails.
        """
        import json, re

        # Try direct JSON parse
        try:
            clean = re.search(r'\{.*?\}', text, re.DOTALL)
            if clean:
                data = json.loads(clean.group())
                return cls(**data)
        except Exception:
            pass

        # Keyword fallback
        priority   = next((p for p in VALID_PRIORITIES   if p in text.upper()), "P3_URGENT")
        department = next((d for d in VALID_DEPARTMENTS  if d in text.upper()), "ER")
        return cls(priority=priority, department=department)


# ── Reward ────────────────────────────────────────────────────────────────────

class TriageReward(BaseModel):
    """Structured reward signal for one triage decision."""

    value:               float = Field(..., ge=-1.0, le=1.0, description="Net reward in [-1, 1]")
    priority_score:      float = Field(..., ge=0.0,  le=1.0, description="Priority accuracy (partial credit)")
    department_score:    float = Field(..., ge=0.0,  le=1.0, description="Routing accuracy")
    resource_penalty:    float = Field(..., ge=0.0,  le=1.0, description="Penalty for routing to full unit")
    under_triage_penalty:float = Field(..., ge=0.0,  le=1.0, description="Penalty for under-triaging critical patients")
    criticality:         float = Field(..., description="Criticality multiplier applied (P1=2x, P2=1.5x, P3=1x, P4=0.8x)")

    true_priority:       str   = Field(..., description="Ground-truth priority")
    true_department:     str   = Field(..., description="Ground-truth department")
    pred_priority:       str   = Field(..., description="Agent's predicted priority")
    pred_department:     str   = Field(..., description="Agent's predicted department")

    @classmethod
    def from_dict(cls, d: dict) -> "TriageReward":
        return cls(
            value                = d["reward"],
            priority_score       = d["priority_score"],
            department_score     = d["department_score"],
            resource_penalty     = d["resource_penalty"],
            under_triage_penalty = d["under_triage_penalty"],
            criticality          = d["criticality"],
            true_priority        = d["true_priority"],
            true_department      = d["true_department"],
            pred_priority        = d["pred_priority"],
            pred_department      = d["pred_department"],
        )


# ── State ─────────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Serialisable environment state returned by state()."""
    task:            str   = Field(..., description="Task name: easy | medium | hard")
    step:            int   = Field(..., description="Current step index")
    n_patients:      int   = Field(..., description="Total patients in episode")
    done:            bool  = Field(..., description="Whether episode is complete")
    icu_beds_left:   int   = Field(..., description="ICU beds remaining")
    er_beds_left:    int   = Field(..., description="ER beds remaining")
    ward_beds_left:  int   = Field(..., description="Ward beds remaining")
    episode_return:  float = Field(..., description="Cumulative reward so far")
    history:         list  = Field(default_factory=list, description="Last 5 step records")


# ── Step Response ─────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    """Full response from env.step(), typed."""
    observation: PatientObservation
    reward:      TriageReward
    done:        bool
    info:        dict