"""
client.py — Python client for MediTriage-Env.

Usage:
    from client import MediTriageClient
    from meditriage_env.schemas import TriageAction

    with MediTriageClient(base_url="https://yatinm-meditriage-env.hf.space").sync() as client:
        result = client.reset(task="medium")
        print(result["observation"])

        result = client.step(action=5)
        print(result["reward"])
"""
from __future__ import annotations
import requests
from typing import Optional


class MediTriageClient:
    """Synchronous HTTP client for MediTriage-Env."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def sync(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    def reset(self, task: str = "medium", seed: Optional[int] = None) -> dict:
        """Reset the environment. Returns initial observation."""
        payload = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        r = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: int) -> dict:
        """Take one step. Returns obs, reward, done, info."""
        r = self._session.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        """Get current environment state."""
        r = self._session.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        return r.json()

    def tasks(self) -> dict:
        """Get task catalog."""
        r = self._session.get(f"{self.base_url}/tasks", timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        """Health check."""
        r = self._session.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"

    print(f"Connecting to {base_url}...")
    with MediTriageClient(base_url=base_url).sync() as client:
        print("Health:", client.health())
        print("Tasks:", client.tasks())

        result = client.reset(task="easy", seed=42)
        print(f"Reset — obs shape: {len(result['observation'])} dims")

        for step in range(3):
            result = client.step(action=5)
            print(f"Step {step+1} — reward: {result['reward']:.3f}, done: {result['done']}")
            if result["done"]:
                break

        print("State:", client.state())
        print("Done ✓")