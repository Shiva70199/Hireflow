from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


IRRELEVANT_ATTRIBUTE_TERMS = {
    "age",
    "young",
    "old",
    "gender",
    "male",
    "female",
    "married",
    "pregnant",
    "religion",
    "race",
    "ethnicity",
}

STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "for",
    "in",
    "on",
    "with",
    "of",
    "at",
    "by",
    "from",
    "is",
    "are",
    "this",
    "that",
}


def normalize_score(score: float) -> float:
    if score <= 0:
        return 0.01
    if score >= 1:
        return 0.99
    return round(score, 3)


def _tokenize(text: str) -> Set[str]:
    tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9\+\#\.]+", text)}
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _overlap_ratio(job_text: str, candidate_text: str) -> float:
    job_tokens = _tokenize(job_text)
    cand_tokens = _tokenize(candidate_text)
    if not job_tokens:
        return 0.0
    return len(job_tokens & cand_tokens) / len(job_tokens)


def score_reasoning_quality(reasoning: str) -> Tuple[float, float, List[str]]:
    details: List[str] = []
    penalty = 0.0
    quality = 1.0
    cleaned = reasoning.strip()

    if not cleaned:
        penalty -= 0.2
        quality = 0.01
        details.append("empty reasoning penalty applied")
        return quality, penalty, details

    if len(cleaned.split()) < 6:
        penalty -= 0.1
        quality = normalize_score(max(0.0, quality - 0.2))
        details.append("reasoning too short, relevance penalty applied")

    reasoning_tokens = _tokenize(cleaned)
    if reasoning_tokens & IRRELEVANT_ATTRIBUTE_TERMS:
        penalty -= 0.1
        quality = normalize_score(max(0.0, quality - 0.2))
        details.append("irrelevant attribute penalty applied")

    return normalize_score(quality), penalty, details


def score_screening(job_text: str, candidate_resume: str, decision: str, reasoning: str) -> Dict:
    overlap = _overlap_ratio(job_text, candidate_resume)
    expected_positive = overlap >= 0.18
    base = 1.0 if (decision == "shortlist") == expected_positive else 0.3
    quality, penalty, details = score_reasoning_quality(reasoning)

    if decision == "reject" and overlap >= 0.30:
        penalty -= 0.15
        details.append("bias check penalty: ignored relevant skills at screening")

    score = normalize_score(max(0.0, min(1.0, (base * 0.7) + (quality * 0.3))))
    return {"score": score, "penalty": penalty, "details": details}


def score_interview(interview_text: str, decision: str, reasoning: str, rating: float | None) -> Dict:
    has_strong_signals = any(
        phrase in interview_text.lower()
        for phrase in ["quantified", "latency", "scalable", "testing", "trade-offs", "monitoring"]
    )
    expected_positive = has_strong_signals
    base = 1.0 if (decision == "shortlist") == expected_positive else 0.35
    if rating is not None:
        rating_alignment = 1.0 - abs(rating - (0.8 if expected_positive else 0.3))
        base = (base * 0.75) + (max(0.0, rating_alignment) * 0.25)

    quality, penalty, details = score_reasoning_quality(reasoning)
    score = normalize_score(max(0.0, min(1.0, (base * 0.7) + (quality * 0.3))))
    return {"score": score, "penalty": penalty, "details": details}


def score_final_decision(
    candidates: List[Dict], ranking: List[int] | None, decision: str, reasoning: str
) -> Dict:
    sorted_truth = sorted(candidates, key=lambda c: c["ground_truth_score"], reverse=True)
    truth_order = [c["id"] for c in sorted_truth]
    details: List[str] = []
    penalty = 0.0

    if decision == "rank" and ranking:
        hits = sum(1 for idx, cid in enumerate(ranking[: len(truth_order)]) if idx < len(truth_order) and cid == truth_order[idx])
        base = hits / max(1, len(truth_order))
    elif decision == "hire":
        best_id = truth_order[0]
        selected_id = ranking[0] if ranking else best_id
        base = 1.0 if selected_id == best_id else 0.4
    else:
        base = 0.3

    quality, reasoning_penalty, reasoning_details = score_reasoning_quality(reasoning)
    details.extend(reasoning_details)
    penalty += reasoning_penalty

    if decision in {"hire", "rank"} and ranking:
        chosen = [c for c in candidates if c["id"] in ranking]
        if chosen:
            max_overlap = max(c["job_overlap"] for c in chosen)
            if max_overlap < 0.12:
                penalty -= 0.15
                details.append("bias check penalty: final choice ignores job-relevant skills")

    score = normalize_score(max(0.0, min(1.0, (base * 0.7) + (quality * 0.3))))
    return {"score": score, "penalty": penalty, "details": details}


def apply_inconsistency_penalty(history: List[str], decision: str) -> Tuple[float, List[str]]:
    details: List[str] = []
    if history and "reject" in history[-1] and decision in {"shortlist", "hire"}:
        details.append("inconsistency penalty: contradictory decision sequence")
        return -0.2, details
    return 0.0, details


def attach_overlap(candidates: List[Dict], job_text: str) -> None:
    for c in candidates:
        c["job_overlap"] = _overlap_ratio(job_text, c["resume_text"])
