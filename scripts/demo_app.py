"""
Gradio demo app for MediTriage-Env on Hugging Face Spaces.
Lets users interactively run the heuristic agent and see triage decisions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import numpy as np
from meditriage_env import MediTriageEnv, action_to_pd
from meditriage_env.models import Priority, Department

_TASK_ENV: dict[str, MediTriageEnv] = {}
_TASK_OBS: dict[str, np.ndarray] = {}
_TASK_LOG: dict[str, list] = {}


def reset_env(task: str, seed: int):
    env = MediTriageEnv(task=task)
    obs = env.reset(seed=int(seed))
    _TASK_ENV[task] = env
    _TASK_OBS[task] = obs
    _TASK_LOG[task] = []
    return _render_patient(env, obs), "Episode reset. Use 'Auto-run Heuristic' to step through."


def _render_patient(env, obs):
    if env._done or env._step_idx >= len(env._patients):
        return "✅ Episode complete."
    p = env._patients[env._step_idx]
    lines = [
        f"**Patient {p.patient_id + 1} / {len(env._patients)}**",
        f"- Chief complaint: `{p.chief_complaint.replace('_', ' ').title()}`",
        f"- Age: {p.age}y | Gender: {'M' if p.gender==0 else 'F' if p.gender==1 else 'Other'}",
        f"- HR: {p.heart_rate:.0f} bpm | BP: {p.systolic_bp:.0f}/{p.diastolic_bp:.0f} mmHg",
        f"- SpO₂: {p.spo2:.0f}% | Temp: {p.temperature:.1f}°C",
        f"- Pain: {p.pain_score:.1f}/10 | GCS: {p.gcs:.0f}/15",
        f"- Queue: {p.queue_length} | ICU beds: {p.icu_beds_left} | ER beds: {p.er_beds_left} | Ward beds: {p.ward_beds_left}",
    ]
    return "\n".join(lines)


def heuristic_step(task: str):
    env = _TASK_ENV.get(task)
    if env is None or env._done:
        return "Run reset first.", "", ""
    obs = _TASK_OBS[task]

    # Heuristic logic
    hr   = obs[0] * 220; spo2 = obs[3] * 100; sbp = obs[1] * 250
    gcs  = obs[6] * 15;  pain = obs[5] * 10
    icu  = obs[26] * 20 > 0; er = obs[27] * 40 > 0

    from meditriage_env.models import Priority, Department
    from meditriage_env import pd_to_action
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

    action = pd_to_action(prio, dept)
    new_obs, reward, done, info = env.step(action)
    _TASK_OBS[task] = new_obs
    _TASK_LOG[task].append(info)

    decision = f"→ **{prio.name}** | **{dept.name}** | Reward: `{reward:.3f}`"
    true_str  = f"Ground truth: {info['true_priority']} → {info['true_department']}"
    if done:
        ep_ret = info.get("episode_return", 0)
        ep_mean = info.get("episode_mean_rew", 0)
        next_pat = f"✅ Episode complete!\nTotal return: `{ep_ret:.3f}` | Mean reward: `{ep_mean:.3f}`"
    else:
        next_pat = _render_patient(env, new_obs)
    return next_pat, decision, true_str


with gr.Blocks(title="MediTriage-Env Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 MediTriage-Env\nInteractive demo of the medical triage OpenEnv environment.")

    with gr.Row():
        task_dd  = gr.Dropdown(["easy","medium","hard"], value="medium", label="Task")
        seed_num = gr.Number(value=42, label="Seed", precision=0)
        reset_btn = gr.Button("🔄 Reset Episode", variant="primary")

    patient_md  = gr.Markdown("Press Reset to start.")
    status_md   = gr.Markdown("")
    decision_md = gr.Markdown("")
    truth_md    = gr.Markdown("")
    step_btn    = gr.Button("▶ Heuristic Step")

    reset_btn.click(reset_env, inputs=[task_dd, seed_num], outputs=[patient_md, status_md])
    step_btn.click(heuristic_step, inputs=[task_dd], outputs=[patient_md, decision_md, truth_md])

    gr.Markdown("---\n*MediTriage-Env v1.0 — OpenEnv medical triage benchmark*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
