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

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_CHAT_JSON_INSTRUCTIONS = """
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

_SYMPTOM_JSON_INSTRUCTIONS = """
Respond with a single JSON object only, matching exactly this shape:
{
  "assessment": "string — a short plain-language summary for the patient",
  "urgency": "green" | "yellow" | "red",
  "possible_conditions": [
    {
      "name": "string",
      "likelihood": "low" | "moderate" | "high",
      "reasoning": "string — why this fits the reported symptoms"
    }
  ],
  "recommended_action": "self_care" | "see_doctor_soon" | "urgent_care" | "emergency",
  "recommended_specialty": "string or null",
  "red_flags": ["string", ...],
  "follow_up_questions": ["string", ...],
  "emergency_flag": true | false
}

Order possible_conditions with the most likely first, and list at most five.
Never state a condition as confirmed — these are possibilities to discuss with
a doctor, not a diagnosis.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def _load_prompt(filename: str, json_instructions: str) -> str:
    base = (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
    return base + "\n" + json_instructions


def _role_for_gemini(role: str) -> str:
    return "model" if role == "ai" else "user"


def _generate_json(system_prompt: str, contents: list[types.Content]) -> dict:
    """Single retry-hardened call path shared by every LLM-backed feature."""
    model = os.getenv("MODEL_NAME", "gemini-flash-lite-latest")
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
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


def get_llm_response(message: str, conversation_history: list[ChatMessage]) -> dict:
    contents = [
        types.Content(role=_role_for_gemini(turn.role), parts=[types.Part.from_text(text=turn.content)])
        for turn in conversation_history
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    system_prompt = _load_prompt("medical_system_prompt.txt", _CHAT_JSON_INSTRUCTIONS)
    return _generate_json(system_prompt, contents)


# Naming the output language in the *system* instruction is what actually
# holds. With the directive only in the user content, a fully-Arabic report
# still came back in English every time — the long English system prompt won.
# The directive is written in English on purpose, even when the answer must be
# Arabic: the model follows an English meta-instruction far more reliably than
# an Arabic one, so long as it names the target language explicitly.
_LANGUAGE_RULE = {
    "en": "CRITICAL: every string value in your JSON response must be written in English.",
    "ar": (
        "CRITICAL: every string value in your JSON response must be written in "
        "Arabic (العربية). Do not answer in English under any circumstances."
    ),
}


def get_symptom_assessment(intake: str, language: str = "en") -> dict:
    """`intake` is the structured symptom report rendered as text — see
    services/symptom_intake.py for how it's built from the request.
    `language` is decided in code ('en' or 'ar'), never inferred by the model."""
    rule = _LANGUAGE_RULE.get(language, _LANGUAGE_RULE["en"])
    base = _load_prompt("symptom_checker_prompt.txt", _SYMPTOM_JSON_INSTRUCTIONS)
    system_prompt = f"{rule}\n\n{base}\n\n{rule}"

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=intake)])]
    return _generate_json(system_prompt, contents)
