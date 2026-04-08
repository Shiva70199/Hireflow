from __future__ import annotations

import json
import os
from typing import Dict, List

from openai import OpenAI
from dotenv import load_dotenv

from app.env import HireFlowEnv
from app.tasks import DEFAULT_TASK_ORDER

# Load local .env automatically for easier execution.
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


def _build_prompt(obs: Dict) -> str:
    return (
        "You are acting as a hiring pipeline agent.\n"
        f"Stage: {obs['stage']}\n"
        f"Job: {obs['job_description']}\n"
        f"Candidate Resume: {obs['candidate_resume']}\n"
        f"Interview Data: {obs.get('interview_data')}\n"
        "Return JSON with keys decision, reasoning, rating (optional), ranking (optional).\n"
        "Decision must be one of shortlist, reject, hire, rank.\n"
    )


def _heuristic_action(obs: Dict, env_state: Dict) -> Dict:
    stage = obs["stage"]
    if stage == "screening":
        resume = str(obs["candidate_resume"]).lower()
        positive_tokens = ["python", "kubernetes", "react", "postgresql", "kafka", "aws", "testing"]
        match = sum(token in resume for token in positive_tokens)
        if match >= 2:
            return {
                "decision": "shortlist",
                "reasoning": "Resume shows relevant skills and measurable delivery impact aligned with job requirements.",
                "rating": 0.82,
            }
        return {"decision": "reject", "reasoning": "Profile does not sufficiently match required technical depth.", "rating": 0.35}

    if stage == "interview":
        interview = (obs.get("interview_data") or "").lower()
        strong = any(k in interview for k in ["quantified", "latency", "trade-offs", "monitoring", "testing"])
        if strong:
            return {"decision": "shortlist", "reasoning": "Interview demonstrates strong technical depth and execution quality.", "rating": 0.86}
        return {"decision": "reject", "reasoning": "Interview feedback indicates gaps in required technical rigor.", "rating": 0.38}

    candidates = env_state.get("candidate_ids", [])
    ranked = candidates[:]
    return {
        "decision": "rank",
        "reasoning": "Ranking based on strongest role alignment, interview strength, and execution outcomes.",
        "ranking": ranked,
        "rating": 0.8,
    }


def _ask_llm(client: OpenAI, model_name: str, obs: Dict) -> Dict:
    prompt = _build_prompt(obs)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise hiring evaluator that returns strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content.strip()
    return json.loads(text)


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _fmt_reward(value: float) -> str:
    return f"{value:.2f}"


def _fmt_action(action: Dict) -> str:
    return json.dumps(action, ensure_ascii=True, separators=(",", ":"))


def run_task(task_name: str) -> None:
    env = HireFlowEnv(task=task_name, seed=42)
    obs_model = env.reset()
    obs = obs_model.model_dump()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    rewards: List[float] = []
    success = True
    last_action_error: str | None = None

    print(f"[START] task={task_name} env=hireflow-multiagent-env model={MODEL_NAME}")
    done = False
    steps = 0

    try:
        while not done:
            action: Dict
            try:
                action = _ask_llm(client, MODEL_NAME, obs)
                last_action_error = None
            except Exception as llm_err:
                action = _heuristic_action(obs, env.state())
                last_action_error = str(llm_err)

            try:
                obs_model, reward_model, done, _info = env.step(action)
                obs = obs_model.model_dump()
                reward = float(reward_model.total)
                rewards.append(reward)
            except Exception as step_err:
                reward = 0.0
                done = True
                success = False
                last_action_error = str(step_err)
                rewards.append(reward)

            steps += 1
            print(
                f"[STEP] step={steps} action={_fmt_action(action)} "
                f"reward={_fmt_reward(reward)} done={_fmt_bool(done)} "
                f"error={last_action_error if last_action_error is not None else 'null'}"
            )

            if steps > env.max_steps + 1:
                success = False
                done = True
                last_action_error = "max_step_guard_triggered"
    finally:
        try:
            env.close()
        finally:
            final_score = rewards[-1] if rewards else 0.0
            final_score = max(0.0, min(1.0, float(final_score)))
            reward_str = ",".join(_fmt_reward(r) for r in rewards)
            print(
                f"[END] success={_fmt_bool(success)} steps={steps} "
                f"score={_fmt_reward(final_score)} rewards={reward_str}"
            )


def main() -> None:
    for task in DEFAULT_TASK_ORDER:
        run_task(task)


if __name__ == "__main__":
    main()
