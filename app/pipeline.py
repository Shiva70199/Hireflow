from __future__ import annotations

from typing import Dict, List


STAGE_FLOW = {
    "screening": "interview",
    "interview": "hr_decision",
    "hr_decision": "done",
}


def next_stage(current_stage: str) -> str:
    return STAGE_FLOW.get(current_stage, "done")


def format_history_entry(stage: str, decision: str, reasoning: str) -> str:
    clean_reasoning = " ".join(reasoning.strip().split())
    return f"{stage}:{decision}:{clean_reasoning}"


def build_state_snapshot(raw_state: Dict) -> Dict:
    return {
        "difficulty": raw_state["difficulty"],
        "stage": raw_state["stage"],
        "step_count": raw_state["step_count"],
        "current_candidate_index": raw_state["current_candidate_index"],
        "screening_score": raw_state["screening_score"],
        "interview_score": raw_state["interview_score"],
        "final_decision_score": raw_state["final_decision_score"],
        "penalties": raw_state["penalties"],
        "history": list(raw_state["history"]),
        "shortlisted_ids": list(raw_state["shortlisted_ids"]),
        "rejected_ids": list(raw_state["rejected_ids"]),
        "hired_ids": list(raw_state["hired_ids"]),
        "candidate_ids": [c["id"] for c in raw_state["candidates"]],
    }


def active_resume_or_list(stage: str, candidates: List[Dict], idx: int):
    if stage == "hr_decision":
        return [c["resume_text"] for c in candidates]
    return candidates[idx]["resume_text"]
