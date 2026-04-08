# HireFlow MultiAgent Env — Complete Runbook (Run / Test / Deploy / Submit)

This project is a **submission-ready OpenEnv environment + SaaS web app**.

You can use it in three ways:
- **Web UI**: a simple hiring-pipeline dashboard
- **HTTP API**: session-based environment control
- **Inference benchmark**: `inference.py` runs tasks and prints strict OpenEnv challenge logs

---

## 1) Start the app (local)

Open PowerShell and run:

```powershell
cd D:\Hackathon\hireflow-multiagent-env
python -m pip install -r requirements.txt
python validate_env.py
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Now open:
- **Web UI**: `http://127.0.0.1:8000/`
- **API Docs**: `http://127.0.0.1:8000/docs`
- **Health**: `http://127.0.0.1:8000/health`

Important (Windows): if `uvicorn` command is not recognized, always use:
`python -m uvicorn ...` (this is expected and correct).

---

## 2) Configure tokens (what goes in `.env`)

File: `.env`

### A) `HF_TOKEN` (required for inference.py only)
This must be your **real Hugging Face token** if you want `inference.py` to call a model via:
`API_BASE_URL=https://router.huggingface.co/v1`

If `HF_TOKEN` is wrong, inference will show `401 Invalid username or password`.

### B) `HIREFLOW_API_TOKEN` (your API password)
This is a password you choose. If set, all API calls must include header:
`x-api-token: <HIREFLOW_API_TOKEN>`

If you don’t want auth locally, you can set it empty.

---

## 3) Web UI testing (recommended for demo)

Go to `http://127.0.0.1:8000/`

### Easy task test
1. Task = `easy`
2. Create session
3. Send step:
   - decision = `shortlist`
   - reasoning = “Strong match for job requirements”
4. Expected: `done=true` in response, reward present.

### Medium task test
1. Task = `medium`
2. Create session
3. Step 1: `shortlist`
4. Step 2: `shortlist`
5. Expected: `done=true` after interview stage.

### Hard task test
1. Task = `hard`
2. Create session
3. Keep sending `shortlist` until stage becomes `hr_decision`
4. Final step:
   - decision = `hire` OR `rank`
   - If `rank`, supply `ranking` like `[1,8,2]`
5. Expected: `done=true`, final reward present.

If you send an invalid action, server will not crash; response includes:
`info.error`.

---

## 4) API testing (OpenEnv-style)

These two endpoints are included for HF Space/OpenEnv server checks:

### POST `/reset`
Creates a new session and returns the initial observation.

Request body:
```json
{"task":"hard","seed":42}
```

Response:
- `session_id`
- `observation`
- `state`

### POST `/step`
Steps the session.

Request body (recommended):
```json
{
  "session_id": "<uuid>",
  "action": {
    "decision": "shortlist",
    "reasoning": "Job-relevant justification",
    "rating": 0.85
  }
}
```

Response always includes:
- `observation`
- `reward`
- `done`
- `info`
- `state`

---

## 5) Benchmark inference (strict hackathon logs)

Run:
```powershell
cd D:\Hackathon\hireflow-multiagent-env
python inference.py
```

It prints exactly:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

If LLM calls fail, it falls back to a deterministic policy, still producing valid logs.

---

## 6) Docker (local + HF Space)

Build:
```powershell
docker build -t hireflow-multiagent-env .
```

Run:
```powershell
docker run --rm -p 8000:8000 ^
  -e HIREFLOW_API_TOKEN=your_api_password ^
  hireflow-multiagent-env
```

Then open:
`http://127.0.0.1:8000/`

---

## 7) Hugging Face Space deployment (Docker SDK)

1. Create a **Docker Space**
2. Push this repo to the Space
3. In Space settings → **Secrets**, add:
   - `HIREFLOW_API_TOKEN` (recommended)
   - `HF_TOKEN` (only needed if you run inference in Space)
4. Wait until Space shows **Running**
5. Verify:
   - `/` loads UI
   - `/health` returns ok
   - `/docs` loads

---

## 8) Submission readiness checklist (quick)

- `python validate_env.py` passes
- server runs and `/reset` + `/step` return JSON successfully
- `inference.py` prints strict logs and includes `score=` in `[END]`
- Docker builds and runs

