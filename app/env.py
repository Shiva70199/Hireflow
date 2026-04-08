from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError
from app.grader import (
    apply_inconsistency_penalty,
    attach_overlap,
    normalize_score,
    score_final_decision,
    score_interview,
    score_screening,
)
from app.models import Action, Observation, Reward
from app.pipeline import active_resume_or_list, build_state_snapshot, format_history_entry, next_stage
from app.tasks import TASK_CONFIGS

ALLOWED_DECISIONS_BY_STAGE: Dict[str, frozenset[str]] = {
    "screening": frozenset({"shortlist", "reject"}),
    "interview": frozenset({"shortlist", "reject"}),
    "hr_decision": frozenset({"hire", "rank"}),
}

def _json_safe(value: Any) -> Any:
    """
    Convert arbitrary objects into JSON-serializable structures.
    Pydantic ValidationError payloads can include non-serializable objects in ctx.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class HireFlowEnv:
    def __init__(self, task: str | None = None, seed: int = 42):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.rng = random.Random(seed)
        self.task = task
        self.max_steps = 8
        self._raw_state: Dict = {}
        self._load_data()

    def _load_data(self) -> None:
        with (self.data_dir / "resumes.json").open("r", encoding="utf-8") as f:
            self.resumes = json.load(f)
        with (self.data_dir / "jobs.json").open("r", encoding="utf-8") as f:
            self.jobs = json.load(f)
        with (self.data_dir / "interviews.json").open("r", encoding="utf-8") as f:
            self.interviews = json.load(f)

    def _build_observation(self) -> Observation:
        stage = self._raw_state["stage"]
        idx = self._raw_state["current_candidate_index"]
        candidates = self._raw_state["candidates"]
        interview_blob = None
        if stage == "interview" and idx < len(candidates):
            interview_blob = self.interviews.get(str(candidates[idx]["id"]), "")
        return Observation(
            stage=stage,
            job_description=self._raw_state["job"]["description"],
            candidate_resume=active_resume_or_list(stage, candidates, min(idx, len(candidates) - 1)),
            interview_data=interview_blob,
            step_count=self._raw_state["step_count"],
            history=list(self._raw_state["history"]),
        )

    def reset(self) -> Observation:
        difficulty = self.task or self.rng.choice(["easy", "medium", "hard"])
        if difficulty not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task difficulty: {difficulty}")
        cfg = TASK_CONFIGS[difficulty]
        self.max_steps = cfg["max_steps"]
        selected_job = self.rng.choice(self.jobs)
        selected_candidates = self.rng.sample(self.resumes, k=cfg["num_candidates"])
        attach_overlap(selected_candidates, selected_job["description"])
        self._raw_state = {
            "difficulty": difficulty,
            "job": selected_job,
            "candidates": selected_candidates,
            "stage": "screening",
            "step_count": 0,
            "current_candidate_index": 0,
            "history": [],
            "screening_score": 0.0,
            "interview_score": 0.0,
            "final_decision_score": 0.0,
            "penalties": 0.0,
            "shortlisted_ids": [],
            "rejected_ids": [],
            "hired_ids": [],
            "done": False,
        }
        return self._build_observation()

    def _normalize_action(self, action) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, dict):
            return Action(**action)
        raise TypeError("action must be an Action or dict")

    def _observation_for_step_return(self) -> Observation:
        if self._raw_state["done"] or self._raw_state["stage"] == "done":
            return Observation(
                stage="hr_decision",
                job_description=self._raw_state["job"]["description"],
                candidate_resume=[c["resume_text"] for c in self._raw_state["candidates"]],
                interview_data=None,
                step_count=self._raw_state["step_count"],
                history=list(self._raw_state["history"]),
            )
        return self._build_observation()

    def _invalid_action_return(
        self, *, error: str, message: str, details: Any = None
    ) -> Tuple[Observation, Reward, bool, Dict]:
        reward = self._compute_total_reward()
        done = bool(self._raw_state["done"])
        obs = self._observation_for_step_return()
        info: Dict[str, Any] = {
            "error": error,
            "message": message,
            "difficulty": self._raw_state.get("difficulty"),
            "details": [],
            "state": build_state_snapshot(self._raw_state),
        }
        if details is not None:
            info["validation_details"] = details
        return obs, reward, done, info

    def _advance_candidate_or_stage(self) -> None:
        idx = self._raw_state["current_candidate_index"] + 1
        if idx < len(self._raw_state["candidates"]):
            self._raw_state["current_candidate_index"] = idx
            return
        current = self._raw_state["stage"]
        difficulty = self._raw_state["difficulty"]
        if difficulty == "easy":
            self._raw_state["stage"] = "done"
            return
        if difficulty == "medium" and current == "interview":
            self._raw_state["stage"] = "done"
            return
        self._raw_state["stage"] = next_stage(current)
        self._raw_state["current_candidate_index"] = 0

    def _compute_total_reward(self) -> Reward:
        total = (
            0.3 * self._raw_state["screening_score"]
            + 0.4 * self._raw_state["interview_score"]
            + 0.3 * self._raw_state["final_decision_score"]
            + self._raw_state["penalties"]
        )
        total = normalize_score(max(0.0, min(1.0, total)))
        return Reward(
            total=round(total, 4),
            screening_score=round(normalize_score(self._raw_state["screening_score"]), 4),
            interview_score=round(normalize_score(self._raw_state["interview_score"]), 4),
            final_decision_score=round(normalize_score(self._raw_state["final_decision_score"]), 4),
            penalties=round(self._raw_state["penalties"], 4),
            details=[],
        )

    def step(self, action) -> Tuple[Observation, Reward, bool, Dict]:
        if not self._raw_state:
            obs = Observation(
                stage="screening",
                job_description="",
                candidate_resume="",
                interview_data=None,
                step_count=0,
                history=[],
            )
            z = Reward(
                total=0.01,
                screening_score=0.01,
                interview_score=0.01,
                final_decision_score=0.01,
                penalties=0.0,
            )
            return obs, z, False, {
                "error": "not_initialized",
                "message": "Environment not initialized. Call reset() first.",
                "details": [],
            }
        if self._raw_state["done"]:
            reward = self._compute_total_reward()
            return self._observation_for_step_return(), reward, True, {"message": "episode already finished"}
        try:
            parsed = self._normalize_action(action)
        except Exception as exc:
            details: Any
            if isinstance(exc, ValidationError):
                details = _json_safe(exc.errors())
            else:
                details = _json_safe(str(exc))
            return self._invalid_action_return(
                error="invalid_action",
                message="Action failed validation. Expected Action dict with valid decision and fields.",
                details=details,
            )
        stage = self._raw_state["stage"]
        allowed = ALLOWED_DECISIONS_BY_STAGE.get(stage)
        if allowed is not None and parsed.decision not in allowed:
            return self._invalid_action_return(
                error="invalid_decision_for_stage",
                message=(
                    f"decision '{parsed.decision}' is not valid for stage '{stage}'. "
                    f"Allowed: {sorted(allowed)}"
                ),
            )
        idx = self._raw_state["current_candidate_index"]
        candidate = self._raw_state["candidates"][min(idx, len(self._raw_state["candidates"]) - 1)]
        self._raw_state["step_count"] += 1
        details: List[str] = []
        inconsistency_penalty, consistency_details = apply_inconsistency_penalty(
            self._raw_state["history"], parsed.decision
        )
        self._raw_state["penalties"] += inconsistency_penalty
        details.extend(consistency_details)
        if stage == "screening":
            result = score_screening(
                self._raw_state["job"]["description"],
                candidate["resume_text"],
                parsed.decision,
                parsed.reasoning,
            )
            self._raw_state["screening_score"] = max(self._raw_state["screening_score"], result["score"])
            self._raw_state["penalties"] += result["penalty"]
            details.extend(result["details"])
            if parsed.decision == "shortlist":
                self._raw_state["shortlisted_ids"].append(candidate["id"])
            else:
                self._raw_state["rejected_ids"].append(candidate["id"])
                if self._raw_state["difficulty"] == "easy":
                    self._raw_state["stage"] = "done"
        elif stage == "interview":
            result = score_interview(
                self.interviews.get(str(candidate["id"]), ""),
                parsed.decision,
                parsed.reasoning,
                parsed.rating,
            )
            self._raw_state["interview_score"] = max(self._raw_state["interview_score"], result["score"])
            self._raw_state["penalties"] += result["penalty"]
            details.extend(result["details"])
            if parsed.decision == "shortlist":
                if candidate["id"] not in self._raw_state["shortlisted_ids"]:
                    self._raw_state["shortlisted_ids"].append(candidate["id"])
            else:
                self._raw_state["rejected_ids"].append(candidate["id"])
                if self._raw_state["difficulty"] == "medium":
                    self._raw_state["stage"] = "done"
        elif stage == "hr_decision":
            result = score_final_decision(
                self._raw_state["candidates"],
                parsed.ranking,
                parsed.decision,
                parsed.reasoning,
            )
            self._raw_state["final_decision_score"] = max(
                self._raw_state["final_decision_score"], result["score"]
            )
            self._raw_state["penalties"] += result["penalty"]
            details.extend(result["details"])
            if parsed.decision in {"hire", "rank"}:
                ranked = parsed.ranking or []
                if ranked:
                    self._raw_state["hired_ids"] = [ranked[0]]
                else:
                    best = sorted(
                        self._raw_state["candidates"],
                        key=lambda c: c["ground_truth_score"],
                        reverse=True,
                    )[0]
                    self._raw_state["hired_ids"] = [best["id"]]
            self._raw_state["stage"] = "done"
        self._raw_state["history"].append(format_history_entry(stage, parsed.decision, parsed.reasoning))
        if self._raw_state["stage"] != "done":
            self._advance_candidate_or_stage()
        if self._raw_state["step_count"] >= self.max_steps:
            self._raw_state["done"] = True
            self._raw_state["stage"] = "done"
            details.append("max steps reached")
        elif self._raw_state["stage"] == "done":
            self._raw_state["done"] = True
        reward = self._compute_total_reward()
        info = {
            "difficulty": self._raw_state["difficulty"],
            "details": details,
            "state": build_state_snapshot(self._raw_state),
        }
        obs = (
            self._build_observation()
            if not self._raw_state["done"]
            else Observation(
                stage="hr_decision",
                job_description=self._raw_state["job"]["description"],
                candidate_resume=[c["resume_text"] for c in self._raw_state["candidates"]],
                interview_data=None,
                step_count=self._raw_state["step_count"],
                history=list(self._raw_state["history"]),
            )
        )
        return obs, reward, self._raw_state["done"], info

    def state(self) -> Dict:
        if not self._raw_state:
            return {}
        return build_state_snapshot(self._raw_state)

    def close(self) -> None:
        return None
