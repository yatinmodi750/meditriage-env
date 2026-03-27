"""
Agent graders for MediTriage-Env.

Each grader runs N episodes and returns a score in [0.0, 1.0].

  easy_grader   — rewards correct priority classification
  medium_grader — rewards combined priority + routing
  hard_grader   — rewards correct triage under resource constraints,
                  with bonus for saving critical patients correctly
"""
from __future__ import annotations
import numpy as np
from meditriage_env import MediTriageEnv, Priority, action_to_pd, Department


def _run_episodes(env: MediTriageEnv, agent_fn, n_episodes: int, seed: int) -> list[float]:
    """Run agent_fn for n_episodes, return per-episode mean rewards."""
    returns = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        rewards = []
        while not done:
            action = agent_fn(obs, env.state())
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
        returns.append(float(np.mean(rewards)))
    return returns


# ── Easy grader ───────────────────────────────────────────────────────────────

def easy_grader(agent_fn, n_episodes: int = 10, seed: int = 0) -> dict:
    """
    Score in [0, 1].
    Measures mean priority-classification accuracy over easy episodes.
    """
    env = MediTriageEnv(task="easy")
    correct_priority = 0
    total = 0

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent_fn(obs, env.state())
            pred_p, _ = action_to_pd(action)
            true_p    = env._patients[env._step_idx].true_priority
            if pred_p == true_p:
                correct_priority += 1
            total += 1
            obs, _, done, _ = env.step(action)

    accuracy = correct_priority / max(total, 1)
    # Map: random baseline ~0.25, perfect = 1.0 → normalise to [0, 1]
    score = float(np.clip((accuracy - 0.25) / 0.75, 0.0, 1.0))

    return {
        "task":             "easy",
        "score":            round(score, 4),
        "priority_accuracy": round(accuracy, 4),
        "episodes":         n_episodes,
        "description":      "Priority classification accuracy (normalised above random).",
    }


# ── Medium grader ─────────────────────────────────────────────────────────────

def medium_grader(agent_fn, n_episodes: int = 10, seed: int = 100) -> dict:
    """
    Score in [0, 1].
    Measures mean episode reward over medium task.
    Baseline: random agent ≈ -0.1, perfect agent ≈ 0.85.
    """
    env = MediTriageEnv(task="medium")
    returns = _run_episodes(env, agent_fn, n_episodes, seed)
    mean_return = float(np.mean(returns))

    # Empirical range: random ≈ -0.1, perfect ≈ 0.85
    lo, hi = -0.10, 0.85
    score = float(np.clip((mean_return - lo) / (hi - lo), 0.0, 1.0))

    return {
        "task":         "medium",
        "score":        round(score, 4),
        "mean_return":  round(mean_return, 4),
        "episodes":     n_episodes,
        "description":  "Mean episode reward (normalised to [0,1] above random baseline).",
    }


# ── Hard grader ───────────────────────────────────────────────────────────────

def hard_grader(agent_fn, n_episodes: int = 10, seed: int = 200) -> dict:
    """
    Score in [0, 1].
    Composite: mean reward + critical-patient accuracy bonus + resource penalty.
    """
    env = MediTriageEnv(task="hard")

    mean_rewards    = []
    critical_correct = 0
    critical_total   = 0
    resource_viol    = 0
    total_steps      = 0

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent_fn(obs, env.state())
            idx    = env._step_idx
            p      = env._patients[idx]

            pred_p, pred_d = action_to_pd(action)

            # Critical patient tracking
            if p.true_priority in (Priority.P1_IMMEDIATE, Priority.P2_EMERGENT):
                critical_total += 1
                if pred_p == p.true_priority and pred_d == p.true_department:
                    critical_correct += 1

            # Resource violation tracking
            if (pred_d == Department.ICU   and env._icu_beds   == 0 or
                pred_d == Department.ER    and env._er_beds    == 0 or
                pred_d == Department.WARD  and env._ward_beds  == 0):
                resource_viol += 1

            total_steps += 1
            obs, reward, done, info = env.step(action)
            mean_rewards.append(reward)

    mean_rew       = float(np.mean(mean_rewards))
    critical_acc   = critical_correct / max(critical_total, 1)
    viol_rate      = resource_viol / max(total_steps, 1)

    lo, hi = -0.15, 0.85
    base_score     = float(np.clip((mean_rew - lo) / (hi - lo), 0.0, 1.0))
    bonus          = 0.15 * critical_acc
    penalty        = 0.10 * viol_rate
    score          = float(np.clip(base_score + bonus - penalty, 0.0, 1.0))

    return {
        "task":              "hard",
        "score":             round(score, 4),
        "mean_return":       round(mean_rew, 4),
        "critical_accuracy": round(critical_acc, 4),
        "resource_viol_rate": round(viol_rate, 4),
        "episodes":          n_episodes,
        "description":       (
            "Composite: mean reward + 15% critical-patient bonus "
            "– 10% resource-violation penalty."
        ),
    }


# ── Master grader ─────────────────────────────────────────────────────────────

def grade_all(agent_fn, n_episodes: int = 10) -> dict:
    """Run all three graders and return overall score."""
    easy   = easy_grader(agent_fn,   n_episodes=n_episodes)
    medium = medium_grader(agent_fn, n_episodes=n_episodes)
    hard   = hard_grader(agent_fn,   n_episodes=n_episodes)

    # Weighted overall: easy=0.2, medium=0.3, hard=0.5
    overall = 0.2 * easy["score"] + 0.3 * medium["score"] + 0.5 * hard["score"]

    return {
        "overall_score": round(overall, 4),
        "easy":          easy,
        "medium":        medium,
        "hard":          hard,
    }
