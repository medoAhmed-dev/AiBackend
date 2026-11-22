import json
import logging
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
_PRIMARY_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 1

# Retrying one model isn't enough: a model can be unavailable for a sustained
# stretch, not just a moment. This has taken the service down three separate
# times, each time with a different model overloaded while others answered
# fine. The "-latest" aliases are the worst affected — they're everyone's
# default, so they absorb the most load.
#
# So after the primary is exhausted, fall through to other models. Each
# fallback gets a single attempt: the point is to find one that's answering,
# not to wait out one that isn't.
_FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

logger = logging.getLogger("medical_chat.llm")


def _models_to_try() -> list[str]:
    primary = os.getenv("MODEL_NAME", _FALLBACK_MODELS[0])
    return [primary] + [m for m in _FALLBACK_MODELS if m != primary]

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

_PREVISIT_JSON_INSTRUCTIONS = """
Respond with a single JSON object only, matching exactly this shape:
{
  "summary": "string — the brief itself, a short paragraph for the doctor",
  "key_points": ["string", ...],
  "questions_for_doctor": ["string", ...],
  "flags_for_clinician": ["string", ...]
}

Leave an array empty rather than padding it. flags_for_clinician is for
things that would change the clinician's approach — red-flag symptoms,
relevant history, allergies — not for routine detail.
"""

_QA_JSON_INSTRUCTIONS = """
Respond with a single JSON object only, matching exactly this shape:
{
  "answer": "string — a direct answer first, then the detail behind it",
  "key_points": ["string", ...],
  "needs_assessment": true | false,
  "related_questions": ["string", ...]
}

Set needs_assessment to true when the question is really about symptoms the
person is experiencing right now, rather than a general question — the app
uses it to steer them to a proper symptom assessment.
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
    """Single retry-hardened call path shared by every LLM-backed feature.

    Retries the configured model, then falls through to alternatives if it
    stays unavailable — a 503 from one model says nothing about the others.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
    )

    models = _models_to_try()
    last_error: Exception | None = None

    for index, model in enumerate(models):
        is_primary = index == 0
        attempts = _PRIMARY_ATTEMPTS if is_primary else 1

        for attempt in range(attempts):
            try:
                response = _get_client().models.generate_content(
                    model=model, contents=contents, config=config
                )
                if not is_primary:
                    logger.warning(
                        "Primary model %s unavailable; served by fallback %s",
                        models[0],
                        model,
                    )
                return json.loads(response.text)
            except errors.ServerError as exc:
                # Transient (5xx, e.g. 503 "high demand").
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(_RETRY_BACKOFF_SECONDS)

    logger.error("Every model was unavailable: %s", ", ".join(models))
    raise last_error


def get_llm_response(
    message: str, conversation_history: list[ChatMessage], language: str = "en"
) -> dict:
    """`language` is decided in code ('en' or 'ar'), never inferred by the
    model. Left to infer, a one-word English message ("headache") came back
    in Arabic 4 times out of 5 — short messages give it too little to go on."""
    contents = [
        types.Content(role=_role_for_gemini(turn.role), parts=[types.Part.from_text(text=turn.content)])
        for turn in conversation_history
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    rule = _LANGUAGE_RULE.get(language, _LANGUAGE_RULE["en"])
    base = _load_prompt("medical_system_prompt.txt", _CHAT_JSON_INSTRUCTIONS)
    return _generate_json(f"{rule}\n\n{base}\n\n{rule}", contents)


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


def _with_language(rule: str, base: str) -> str:
    """Language rule stated at both ends: a trailing instruction carries more
    weight than one buried above a long prompt, and the leading one sets the
    frame before the model reads anything else."""
    separator = "\n\n"
    return separator.join([rule, base, rule])


def get_previsit_summary(intake: str, language: str = "en") -> dict:
    """`intake` is the appointment context rendered as text. `language` is
    decided in code ('en' or 'ar'), never inferred by the model."""
    rule = _LANGUAGE_RULE.get(language, _LANGUAGE_RULE["en"])
    base = _load_prompt("previsit_summary_prompt.txt", _PREVISIT_JSON_INSTRUCTIONS)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=intake)])]
    return _generate_json(_with_language(rule, base), contents)


def get_medical_answer(question: str, language: str = "en") -> dict:
    """A general medical question, not a symptom report."""
    rule = _LANGUAGE_RULE.get(language, _LANGUAGE_RULE["en"])
    base = _load_prompt("medical_qa_prompt.txt", _QA_JSON_INSTRUCTIONS)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    return _generate_json(_with_language(rule, base), contents)
