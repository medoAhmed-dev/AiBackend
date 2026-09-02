import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from schemas.chat import ChatMessage

# Gemini's own client already retries internally, but during a real demand
# spike that's not always enough (observed firsthand: 503 UNAVAILABLE /
# "high demand", intermittently, even after its internal retry gave up).
# One extra layer of backoff here measurably improves success odds without
# making a failing request hang for too long.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (1, 2)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "medical_system_prompt.txt"

_JSON_INSTRUCTIONS = """
Respond with a single JSON object only, matching exactly this shape:
{
  "reply": "string — natural language response to show the patient",
  "urgency": "green" | "yellow" | "red",
  "possible_causes": ["string", ...],
  "recommended_specialty": "string or null",
  "suggested_tests": ["string", ...],
  "follow_up_questions": ["string", ...],
  "emergency_flag": true | false
}
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8") + "\n" + _JSON_INSTRUCTIONS


def _role_for_gemini(role: str) -> str:
    return "model" if role == "ai" else "user"


def get_llm_response(message: str, conversation_history: list[ChatMessage]) -> dict:
    contents = [
        types.Content(role=_role_for_gemini(turn.role), parts=[types.Part.from_text(text=turn.content)])
        for turn in conversation_history
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    model = os.getenv("MODEL_NAME", "gemini-flash-latest")
    config = types.GenerateContentConfig(
        system_instruction=_load_system_prompt(),
        response_mime_type="application/json",
    )

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = _get_client().models.generate_content(
                model=model, contents=contents, config=config
            )
            return json.loads(response.text)
        except errors.ServerError as exc:
            # Transient (5xx, e.g. 503 "high demand") — worth a retry.
            last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS[attempt])

    raise last_error
