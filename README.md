---
title: HireFlow-MultiAgentEnv
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - evaluation
---

# HireFlow-MultiAgentEnv

`HireFlow-MultiAgentEnv` is an OpenEnv-compliant environment that simulates a realistic multi-stage hiring workflow where an AI agent acts as:
- Resume Screener
- Technical Interview Evaluator
- HR Decision Maker

It also ships as a lightweight SaaS API service using FastAPI, so you can create sessions and run hiring episodes over HTTP.

## Problem
Hiring teams need consistent, explainable, and fair decision processes. This environment benchmarks agents on pipeline decisions across resume screening, interview evaluation, and final selection.

## Why This Matters
- Reflects real-world hiring automation complexity
- Tests multi-stage reasoning instead of single-step classification
- Includes fairness penalties to discourage irrelevant-attribute reasoning
- Supports deterministic reward scoring for reproducible evaluations

## Multi-Agent Pipeline
Stages:
1. `screening`
2. `interview`
3. `hr_decision`

Transition rules:
- Screening reject -> episode ends (easy) or candidate removed
- Screening shortlist -> interview
- Interview shortlist/pass -> HR decision path
- HR decision completes episode

## OpenEnv Interface
Implemented in `app/env.py` via class `HireFlowEnv`:
- `reset()`
- `step(action)`
- `state()`

## Observation Space
Pydantic model: `Observation` in `app/models.py`
- `stage: str`
- `job_description: str`
- `candidate_resume: str | list[str]`
- `interview_data: optional str`
- `step_count: int`
- `history: list[str]`

## Action Space
Pydantic model: `Action` in `app/models.py`
- `decision: str` (`shortlist`, `reject`, `hire`, `rank`)
- `reasoning: str`
- `rating: optional float [0,1]`
- `ranking: optional list[int]`

## Reward Design
Pydantic model: `Reward` in `app/models.py`

Weighted objective:
- `screening_score * 0.3`
- `interview_score * 0.4`
- `final_decision_score * 0.3`

Total reward is clamped to `[0,1]`.

Deterministic grader functions in `app/grader.py`:
- `score_screening()`
- `score_interview()`
- `score_final_decision()`
- `score_reasoning_quality()`

Penalties:
- Empty reasoning: `-0.2`
- Irrelevant/too-short reasoning: up to `-0.1`
- Inconsistent decisions: `-0.2`

## Fairness / Bias Penalty
Bias-aware penalties are applied when:
- Decisions ignore relevant job skills (e.g., reject with high overlap)
- Reasoning references irrelevant personal attributes (age, gender, religion, etc.)

These penalties directly reduce total reward.

## Tasks
- **easy**: single candidate, screening only
- **medium**: screening + interview
- **hard**: multi-candidate full pipeline with ranking and selection

## Data
Located under `app/data/`:
- `resumes.json` (8 realistic profiles)
- `jobs.json` (4 job descriptions)
- `interviews.json` (candidate interview summaries)

## Setup
```bash
pip install -r requirements.txt
```

Create `.env` from sample:
```bash
cp .env.example .env
```

## Run As SaaS API
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Web dashboard (browser UI):
- `http://localhost:8000/`

Interactive API docs:
- `http://localhost:8000/docs`

## Full Guide (Run/Test/Deploy/Submit)
See `RUNBOOK.md`.

## Product Overview (Human-Friendly)
See `PRODUCT_BRIEF.md`.

Optional auth:
- set `HIREFLOW_API_TOKEN`
- send header `x-api-token: <token>` on every API request

### Core API Endpoints
- `GET /health` - service health
- `POST /v1/sessions` - create hiring episode
- `POST /v1/sessions/{session_id}/step` - submit action
- `GET /v1/sessions/{session_id}/state` - fetch state
- `DELETE /v1/sessions/{session_id}` - close session

### Example API Flow (curl)
```bash
# Create session
curl -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"hard\",\"seed\":42}"

# Step (replace SESSION_ID)
curl -X POST http://localhost:8000/v1/sessions/SESSION_ID/step \
  -H "Content-Type: application/json" \
  -d "{\"decision\":\"shortlist\",\"reasoning\":\"Candidate has strong role-relevant skills.\",\"rating\":0.83}"
```

## Validate OpenEnv
```bash
openenv validate
```

If `openenv` CLI is unavailable in your shell, run the local contract validator:
```bash
python validate_env.py
```

## Run Inference
Set environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Then run:
```bash
python inference.py
```

Default inference endpoint is Hugging Face router:
- `API_BASE_URL=https://router.huggingface.co/v1`
- OpenAI client is used for OpenAI-compatible API calls as required.

Inference prints strict logs:
- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... rewards=...`

## Example Output
```text
[START] task=easy env=hireflow-multiagent-env model=gpt-4.1-mini
[STEP] step=1 action={"decision":"shortlist","reasoning":"Resume strongly matches role requirements.","rating":0.84} reward=0.30 done=true error=null
[END] success=true steps=1 rewards=0.30
```

## Baseline Performance (Seed=42)
- `easy`: rewards trajectory `0.30`
- `medium`: rewards trajectory `0.30,0.70`
- `hard`: rewards trajectory `0.30,0.30,0.30,0.50,0.50,0.50,0.66`

## Docker
Build:
```bash
docker build -t hireflow-multiagent-env .
```

Run:
```bash
docker run --rm -p 8000:8000 -e HIREFLOW_API_TOKEN=change-me hireflow-multiagent-env
```

Run inference in container instead of API server:
```bash
docker run --rm -e API_BASE_URL=https://router.huggingface.co/v1 -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct -e HF_TOKEN=your_token hireflow-multiagent-env python inference.py
```

## Build Release Zip
```bash
python make_release.py
```
Artifact path:
`dist/hireflow-multiagent-env.zip`

## Performance Notes
- No heavy ML models are loaded in-process
- Small JSON datasets
- Deterministic scoring and bounded episode length
- Designed to run comfortably within 8 GB memory and under 20 minutes
