from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def main() -> None:
    root = Path(__file__).resolve().parent
    openenv_file = root / "openenv.yaml"
    _assert(openenv_file.exists(), "openenv.yaml not found")

    data = yaml.safe_load(openenv_file.read_text(encoding="utf-8"))
    for key in ("name", "description", "entrypoint", "tasks", "metadata"):
        _assert(key in data, f"Missing key in openenv.yaml: {key}")

    entrypoint = data["entrypoint"]
    _assert(":" in entrypoint, "entrypoint must be in module:Class format")
    module_name, class_name = entrypoint.split(":", 1)

    module = importlib.import_module(module_name)
    env_cls: Any = getattr(module, class_name)
    env = env_cls(task="hard", seed=42)

    _assert(hasattr(env, "reset"), "Environment missing reset()")
    _assert(hasattr(env, "step"), "Environment missing step(action)")
    _assert(hasattr(env, "state"), "Environment missing state()")

    obs = env.reset()
    _assert(hasattr(obs, "model_dump"), "reset() should return a Pydantic model")

    action = {
        "decision": "shortlist",
        "reasoning": "Candidate has role-relevant skills and measurable impact.",
        "rating": 0.8,
    }
    obs2, reward, done, info = env.step(action)
    _assert(hasattr(obs2, "model_dump"), "step() should return Observation model first")
    _assert(hasattr(reward, "model_dump"), "step() should return Reward model second")
    _assert(isinstance(done, bool), "step() should return bool done as third value")
    _assert(isinstance(info, dict), "step() should return dict info as fourth value")

    snapshot = env.state()
    _assert(isinstance(snapshot, dict), "state() should return dict")

    print("Validation successful: environment contract is satisfied.")


if __name__ == "__main__":
    main()
