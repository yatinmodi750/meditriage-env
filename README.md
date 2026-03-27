# 🏥 MediTriage-Env

**A real-world OpenEnv environment for medical triage AI agents.**

An agent must simultaneously assign **priority levels (P1–P4)** and **route patients to departments (ICU / ER / Ward / Discharge)** for a stream of incoming patients — balancing clinical accuracy with resource constraints.

---

## Overview

| | |
|---|---|
| **Observation space** | `Box(29,)` — vitals, demographics, complaint, queue context |
| **Action space** | `Discrete(16)` — (4 priorities) × (4 departments) |
| **Reward** | `[-1.0, 1.0]` — balanced accuracy + routing + criticality |
| **Tasks** | `easy` (20 patients) · `medium` (50) · `hard` (80) |
| **API** | `step()` / `reset()` / `state()` |

---

## Quick Start

```bash
git clone https://github.com/YOUR_USER/meditriage-env
cd meditriage-env
pip install -e .
python scripts/baseline_inference.py
```

---

## Environment API

```python
from meditriage_env import MediTriageEnv

env = MediTriageEnv(task="medium")   # "easy" | "medium" | "hard"
obs = env.reset(seed=42)             # → np.ndarray shape (29,)

done = False
while not done:
    action = your_agent(obs)         # int in [0, 15]
    obs, reward, done, info = env.step(action)
    print(env.render())

state = env.state()                  # serialisable state dict
```

---

## Action Space

`action = priority_index * 4 + department_index`

| Index | Priority | Meaning |
|-------|----------|---------|
| 0 | P1_IMMEDIATE | Life-threatening — act within minutes |
| 1 | P2_EMERGENT | High-risk — act within 15 min |
| 2 | P3_URGENT | Stable but needs care <1 hr |
| 3 | P4_NON_URGENT | Minor — can wait 2+ hrs |

| Index | Department | Meaning |
|-------|-----------|---------|
| 0 | ICU | Intensive Care Unit |
| 1 | ER | Emergency Room |
| 2 | WARD | General Ward |
| 3 | DISCHARGE | Safe to discharge |

Example: `action=4` → P2_EMERGENT + ICU

Helper utilities:

```python
from meditriage_env import action_to_pd, pd_to_action
priority, department = action_to_pd(5)   # → P2_EMERGENT, ER
action = pd_to_action(Priority.P1_IMMEDIATE, Department.ICU)   # → 0
```

---

## Observation Space

29-dimensional `float32` vector (all values normalised to [0, 1]):

| Indices | Feature | Normalisation |
|---------|---------|--------------|
| 0 | Heart rate | ÷ 220 |
| 1 | Systolic BP | ÷ 250 |
| 2 | Diastolic BP | ÷ 150 |
| 3 | SpO₂ | ÷ 100 |
| 4 | Temperature | ÷ 42 |
| 5 | Pain score | ÷ 10 |
| 6 | GCS | ÷ 15 |
| 7 | Age | ÷ 100 |
| 8 | Gender | 0=M, 0.5=F, 1=Other |
| 9–23 | Chief complaint (one-hot) | 15 complaints |
| 24 | Arrival time | ÷ 480 min |
| 25 | Queue length | ÷ 50 |
| 26 | ICU beds remaining | ÷ 20 |
| 27 | ER beds remaining | ÷ 40 |
| 28 | Ward beds remaining | ÷ 60 |

> ⚠️ In `medium` and `hard` tasks, some vitals may be **missing** (set to 0). Agents must handle incomplete observations.

---

## Reward Function

```
reward = clip(
    (0.45 × priority_score + 0.45 × dept_score
   - 0.05 × resource_penalty - 0.05 × under_triage_penalty)
   × criticality  /  normaliser,
   -1.0, 1.0
)
```

- **Priority score**: partial credit — off by 1 level = 0.67, off by 2 = 0.33
- **Department score**: full compatibility matrix (e.g. ER→ICU gets 0.5)
- **Criticality multiplier**: P1=2×, P2=1.5×, P3=1×, P4=0.8×
- **Under-triage penalty**: extra penalty for sending P1 to Ward/Discharge
- **Resource penalty**: routing to a full unit

---

## Tasks

### 🟢 Easy
- 20 patients per episode
- Clean, complete vitals
- Ample bed capacity
- **Goal**: learn basic priority classification

### 🟡 Medium
- 50 patients per episode
- 10% chance of missing vitals per feature
- Moderate constraints (8 ICU / 20 ER / 30 Ward beds)
- **Goal**: handle noise + resource awareness

### 🔴 Hard
- 80 patients per episode
- 25% missing vitals
- Severe constraints (3 ICU / 8 ER / 15 Ward beds)
- 25% rare critical edge cases
- **Goal**: robust triage under pressure

---

## Graders

```python
from graders.graders import grade_all

def my_agent(obs, state):
    return my_model.predict(obs)

results = grade_all(my_agent, n_episodes=20)
print(results["overall_score"])   # weighted: easy×0.2 + medium×0.3 + hard×0.5
```

Individual graders:

```python
from graders.graders import easy_grader, medium_grader, hard_grader
easy_grader(my_agent)
medium_grader(my_agent)
hard_grader(my_agent)
```

---

## Baseline Scores

| Agent | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Random | ~0.00 | ~0.05 | ~0.00 | ~0.03 |
| Heuristic | ~0.45 | ~0.38 | ~0.30 | ~0.35 |
| Oracle (upper bound) | ~0.95 | ~0.88 | ~0.80 | ~0.85 |

Run baselines yourself:

```bash
python scripts/baseline_inference.py
```

---

## Project Structure

```
meditriage-env/
├── meditriage_env/
│   ├── __init__.py
│   ├── env.py              # MediTriageEnv — main OpenEnv class
│   ├── models.py           # Patient, Priority, Department, StepResult
│   ├── patient_generator.py # Procedural patient synthesis
│   └── reward.py           # Balanced reward function
├── graders/
│   └── graders.py          # easy_grader, medium_grader, hard_grader, grade_all
├── scripts/
│   ├── baseline_inference.py  # Reproducible baseline scores
│   └── demo_app.py            # Gradio demo (Hugging Face Spaces)
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # HF Spaces deployment
├── setup.py
└── README.md
```

---

## Docker / Hugging Face Spaces

```bash
docker build -t meditriage-env .
docker run -p 7860:7860 meditriage-env
```

Then open `http://localhost:7860` for the interactive Gradio demo.

For Hugging Face Spaces:
1. Create a new Space → Docker SDK
2. Push this repo to the Space
3. The demo starts automatically

---

## Citation

```bibtex
@misc{meditriage2024,
  title  = {MediTriage-Env: A Medical Triage OpenEnv Environment},
  year   = {2024},
  note   = {OpenEnv benchmark for medical AI agents}
}
```

---

## License

MIT License. This environment is for research and educational purposes.
Clinical decisions must always be made by qualified medical professionals.
