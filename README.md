# Aether Threads — AI Backend

FastAPI service for the AI features. Step 1 (this README) covers the Medical GPT Chat endpoint.

## Setup

```bash
cd AI
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in real values (`GEMINI_API_KEY` at minimum, plus
`SUPABASE_JWT_SECRET` so token verification works).

## Run

```bash
uvicorn main:app --reload --port 8000
```

## Endpoints

### `POST /chat`

Requires `Authorization: Bearer <supabase_jwt>`.

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

Safety layer (runs before/around the LLM call, not left to model judgment alone):
- A hardcoded red-flag keyword list forces `urgency: "red"` and `emergency_flag: true`,
  short-circuiting to a fixed emergency message — see `services/safety.py`.
- Every reply gets a "not a diagnosis" disclaimer appended if the model didn't include one.
- Messages that look like a request for a specific diagnosis/dosage/prescription are logged
  (not blocked) for later audit.

Stateless: `conversation_history` is passed in by the caller each time. No DB read yet.

## Test

```bash
python -m pytest
```

Covers: mild-symptom (non-emergency) response, hardcoded emergency override, empty-input
rejection, and multi-turn history being forwarded to the model.

## Quick manual check

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer <a real supabase jwt>" \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","message":"I have had a headache and fever for three days.","conversation_history":[]}'
```
