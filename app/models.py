from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Observation(BaseModel):
    stage: str = Field(description="Current stage: screening/interview/hr_decision")
    job_description: str
    candidate_resume: Union[str, List[str]]
    interview_data: Optional[str] = None
    step_count: int
    history: List[str] = Field(default_factory=list)


class Action(BaseModel):
    decision: str = Field(description="shortlist/reject/hire/rank")
    reasoning: str
    rating: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ranking: Optional[List[int]] = None

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, value: str) -> str:
        allowed = {"shortlist", "reject", "hire", "rank"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"decision must be one of {sorted(allowed)}")
        return normalized


class Reward(BaseModel):
    total: float = Field(ge=0.0, le=1.0)
    screening_score: float = Field(ge=0.0, le=1.0)
    interview_score: float = Field(ge=0.0, le=1.0)
    final_decision_score: float = Field(ge=0.0, le=1.0)
    penalties: float = Field(le=0.0)
    details: List[str] = Field(default_factory=list)
