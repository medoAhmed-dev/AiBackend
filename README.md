# Aether Threads — AI Backend

FastAPI service for the project's AI features. Phase 1 is complete: a medical guidance
chat endpoint (LLM-based) and a queue wait-time prediction endpoint (classic ML).

**Live:** https://aibackend-l780.onrender.com

| Endpoint | Method | What it does | Approach |
|---|---|---|---|
| `/chat` | POST | Medical guidance chat (EN + AR) | Gemini API |
| `/predict-queue` | POST | Estimated wait time | XGBoost regressor |
| `/health` | GET | Liveness check | — |
| `/test` | GET | Browser test page for `/chat` | — |

Both `/chat` and `/predict-queue` require `Authorization: Bearer <supabase_jwt>`.

## Setup

```bash
cd AI
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in real values — `GEMINI_API_KEY` and
`SUPABASE_JWT_SECRET` at minimum.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API access for `/chat` |
| `MODEL_NAME` | Gemini model (default `gemini-flash-lite-latest`) |
| `SUPABASE_JWT_SECRET` | Verifies incoming auth tokens |
| `ENVIRONMENT` | `development` locally; **must be `production` when deployed** (see below) |

## Run

```bash
uvicorn main:app --reload --port 8000
```

## Test

```bash
python -m pytest
```

15 tests: 6 for `/chat` behavior, 3 for LLM retry logic, 6 for `/predict-queue`.

---

## `POST /chat`

Stateless — `conversation_history` is passed in by the caller each time (no DB reads;
persistence is Phase 2).

Request:
```json
{
  "patient_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "I have had a headache and fever for three days.",
  "conversation_history": [
    {"role": "patient", "content": "..."},
    {"role": "ai", "content": "..."}
  ]
}
```

Response:
```json
{
  "reply": "string",
  "urgency": "green | yellow | red",
  "possible_causes": ["string"],
  "recommended_specialty": "string or null",
  "suggested_tests": ["string"],
  "follow_up_questions": ["string"],
  "emergency_flag": false
}
```

**Bilingual:** detects Arabic vs. English from the patient's message and replies in the
same language (majority language if mixed).

**Safety layer** (`services/safety.py`) — deliberately not left to the model's judgment:
- A hardcoded red-flag keyword list (English **and** Arabic, including colloquial phrasing)
  forces `urgency: "red"` + `emergency_flag: true` and short-circuits to a fixed emergency
  message **before the LLM is called at all**.
- Every reply gets a "not a diagnosis" disclaimer appended if the model didn't include one,
  in the same language as the reply.
- Messages that look like a request for a specific diagnosis/dosage/prescription are logged
  (not blocked) for later audit.

**Reliability:** `services/llm_client.py` retries transient Gemini 5xx errors (up to 3
attempts, short backoff). Gemini's own client retries internally, but during real demand
spikes that isn't always enough.

---

## `POST /predict-queue`

Request:
```json
{
  "doctor_id": "c3d4e5f6-a7b8-9012-cdef-345678901234",
  "current_queue_length": 5,
  "average_consultation_minutes": 12,
  "time_of_day": "14:30",
  "day_of_week": "Tuesday",
  "emergency_patients_ahead": 0
}
```

Response:
```json
{
  "estimated_wait_minutes": 75,
  "queue_position": 6,
  "confidence": "high",
  "explanation": "Based on current queue length and average consultation time..."
}
```

`confidence` is bucketed by how close the request falls to the training distribution
(both key features within the 5th–95th percentile → `high`, neither → `low`) — it reflects
actual data support, not a made-up number.

### The model

Trained by `models/queue_prediction/train.py`, persisted to `model.pkl` (committed, so the
repo runs without a training step). Retrain with:

```bash
python -m models.queue_prediction.train
```

Held-out test set (80/20 split, 5,000 samples):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| LinearRegression (baseline) | 9.24 min | 12.95 min | 0.864 |
| **XGBoost (production)** | **6.84 min** | **9.43 min** | **0.928** |

Features: queue length, average consultation minutes, minutes since midnight, emergency
patients ahead, one-hot day of week.

### Training data

**Synthetic, not real hospital records** — see [`datasets/README.md`](datasets/README.md)
for the full generation method and rationale. Regenerate with:

```bash
python datasets/generate_synthetic.py
```

This must be stated as synthetic in the graduation report, not presented as real-world data.

---

## Deployment (Render)

Auto-builds from `main`. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

> **`ENVIRONMENT` must be `production` in the deployed environment.** The `/dev/token`
> helper endpoint mints a valid auth token for *any* patient ID using the real JWT secret.
> It is gated behind `ENVIRONMENT=development` and returns 404 otherwise. If this is ever
> left as `development` in production, anyone on the internet can mint valid tokens.

Free tier spins down after ~15 minutes idle; the next request cold-starts (~50s).

## Project layout

```
AI/
├── api/              # Route handlers (chat.py, queue.py)
├── core/auth.py      # Supabase JWT verification (shared dependency)
├── datasets/         # Synthetic queue data + generator + docs
├── models/           # Trained model artifacts and training scripts
├── prompts/          # Medical system prompt (editable, plain text)
├── schemas/          # Pydantic request/response contracts
├── services/         # LLM client, safety layer, queue predictor
├── static/test.html  # Browser test page for /chat
└── tests/
```
