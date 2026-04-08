# HireFlow MultiAgent Env — Product Brief (Human-Friendly)

## What this project is
HireFlow MultiAgent Env is a **real-world hiring pipeline simulator** packaged as:
- an **OpenEnv environment** (for evaluation/benchmarking), and
- a small **SaaS-style web application + API** (so you can run sessions in a browser and show results like a product demo).

It tests how well an AI agent can make consistent, fair, explainable decisions across a realistic multi-stage hiring workflow:
1. **Resume Screener**
2. **Technical Interview Evaluator**
3. **HR Decision Maker**

---

## How it works (simple story)
Think of an “episode” as one hiring run:

1. **Reset**
   - A job description is selected.
   - Candidate(s) are selected (1 for easy/medium, multiple for hard).
   - The pipeline starts at **screening**.

2. **Step loop**
   - The agent submits an action:
     - decision (`shortlist/reject` during screening/interview; `hire/rank` at HR decision)
     - reasoning (mandatory)
     - optional rating and ranking
   - The environment **updates state** and **gives an incremental reward**.
   - The episode ends when the final decision is reached or max steps is hit.

3. **State**
   - At any time you can read the pipeline’s state: stage, scores, shortlist/reject/hire IDs, and history.

---

## What makes it “real-world” (not a toy problem)
This environment simulates tasks humans do daily:
- Reading job requirements and resumes
- Evaluating interview feedback
- Making consistent final hiring decisions
- Documenting reasoning for auditability

---

## How it aligns with your hackathon needs

### OpenEnv compliance
- **`reset()`** returns a typed observation (Pydantic model).
- **`step(action)`** always returns `(observation, reward, done, info)` and does **not crash** on invalid inputs.
- **`state()`** returns the full pipeline state snapshot (stage + scores + decisions + history).
- **`openenv.yaml`** declares environment metadata, tasks, and entrypoint.

### Three tasks (easy → medium → hard)
- **easy**: 1 candidate, resume screening only
- **medium**: screening + interview evaluation
- **hard**: multiple candidates, full pipeline including HR ranking/hire

### Deterministic grading (0.0–1.0)
Scoring is reproducible and bounded in \([0,1]\), using deterministic signals from:
- job/resume skill overlap
- interview evidence signals
- final selection alignment vs ground-truth ordering

### Meaningful reward throughout the run
Reward is incremental, not binary:
- screening contributes
- interview contributes
- final HR decision contributes

### Fairness / bias feature (built-in)
Reward is penalized if:
- reasoning uses irrelevant personal attributes (age/gender/religion/race terms), or
- decisions ignore strong job-relevant evidence.

---

## Advantages (why it’s a strong submission)
- **Realistic workflow**: mirrors hiring operations and decision documentation
- **Multi-stage**: tests consistency across roles and stages
- **Explainable**: reasoning required and stored in history
- **Safety**: invalid actions handled gracefully (no crashes)
- **Deployment-ready**: Docker + HF Space-friendly `/reset` and `/step` endpoints + web UI at `/`

---

## Future scope (how this can evolve)
- **Richer interviews**: question-level rubrics and multiple question rounds
- **Fairness improvements**: configurable policies, stronger detection, audit exports
- **Org constraints**: headcount/budget policies, competing reqs, time-to-hire
- **Multi-agent orchestration**: explicit role handoffs (screener → interviewer → HR) with artifacts
- **Production SaaS features**: user accounts, role-based access, usage metering, analytics dashboards

---

## One test case (example you can run)

This is a full example test you can run from the web UI.

### Start the server

```powershell
cd D:\Hackathon\hireflow-multiagent-env
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Open:
`http://127.0.0.1:8000/`

### Create a hard session
- Task: `hard`
- Seed: `42`
- Click **Create session**

### Screening stage actions (repeat)
While `state.stage` is `screening`, send:
- decision: `shortlist`
- reasoning: `Resume matches job skills and shows measurable impact aligned with the requirements.`
- rating: `0.82`

### Interview stage actions (repeat)
While `state.stage` is `interview`, send:
- decision: `shortlist`
- reasoning: `Interview demonstrates depth, trade-offs, testing, and strong execution outcomes.`
- rating: `0.86`

### HR decision (final)
When `state.stage` becomes `hr_decision`, send:
- decision: `hire`
- reasoning: `Hiring the most aligned candidate based on evidence across resume + interview.`

Expected result:
- `done = true`
- `reward.total` is present (0.00–1.00)
- `state.hired_ids` contains the chosen candidate id

### Invalid input example (proves safety)
If you send decision = `string`, the server returns HTTP 200 with:
- `info.error = "invalid_action"`
- `info.validation_details` explaining what was wrong

