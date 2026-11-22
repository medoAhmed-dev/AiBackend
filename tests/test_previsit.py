from unittest.mock import patch

from fastapi.testclient import TestClient

from core.auth import verify_token
from main import app

app.dependency_overrides[verify_token] = lambda: "test-sub"

client = TestClient(app)

PATIENT_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

MOCK_SUMMARY = {
    "summary": "Patient reports a sore throat and fever over three days.",
    "key_points": ["Sore throat, 3 days", "Fever, moderate"],
    "questions_for_doctor": ["Is a strep test warranted?"],
    "flags_for_clinician": [],
}

MOCK_ANSWER = {
    "answer": "Paracetamol reduces fever by acting on the brain's temperature control.",
    "key_points": ["Works centrally, not on the infection itself"],
    "needs_assessment": False,
    "related_questions": ["How is it different from ibuprofen?"],
}


def _summary(payload: dict):
    return client.post(
        "/previsit-summary", json=payload, headers={"Authorization": "Bearer t"}
    )


def _ask(payload: dict):
    return client.post("/ask", json=payload, headers={"Authorization": "Bearer t"})


# --- pre-visit summary ---------------------------------------------------


@patch("services.llm_client.get_previsit_summary", return_value=MOCK_SUMMARY)
def test_summary_returns_a_brief_for_the_doctor(mock_llm):
    response = _summary(
        {
            "patient_id": PATIENT_ID,
            "reason_for_visit": "Sore throat that is not improving",
            "symptoms": ["sore throat", "fever"],
            "duration_days": 3,
            "severity": "moderate",
        }
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]
    assert body["key_points"]


@patch("services.llm_client.get_previsit_summary", return_value=MOCK_SUMMARY)
def test_red_flags_reach_the_clinician_even_if_the_model_missed_them(mock_llm):
    # The model returned no flags; the patient mentioned chest pain. The
    # clinician must still see it.
    response = _summary(
        {
            "patient_id": PATIENT_ID,
            "reason_for_visit": "Chest pain when climbing stairs",
            "symptoms": ["chest pain"],
        }
    )

    body = response.json()
    assert any("chest pain" in f.lower() for f in body["flags_for_clinician"])


@patch("services.llm_client.get_previsit_summary", return_value=MOCK_SUMMARY)
def test_summary_omits_absent_optional_fields_from_the_intake(mock_llm):
    _summary({"patient_id": PATIENT_ID, "reason_for_visit": "Follow-up visit"})

    intake = mock_llm.call_args[0][0]
    assert "Allergies" not in intake
    assert "Age" not in intake


@patch("services.llm_client.get_previsit_summary", return_value=MOCK_SUMMARY)
def test_arabic_reason_requests_an_arabic_summary(mock_llm):
    _summary({"patient_id": PATIENT_ID, "reason_for_visit": "ألم في الحلق منذ ثلاثة أيام"})

    assert mock_llm.call_args.kwargs["language"] == "ar"


def test_summary_requires_a_reason_for_visit():
    assert _summary({"patient_id": PATIENT_ID, "reason_for_visit": ""}).status_code == 422


# --- general Q&A ---------------------------------------------------------


@patch("services.llm_client.get_medical_answer", return_value=MOCK_ANSWER)
def test_general_question_is_answered(mock_llm):
    response = _ask({"patient_id": PATIENT_ID, "question": "How does paracetamol work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["needs_assessment"] is False


@patch("services.llm_client.get_medical_answer", return_value=MOCK_ANSWER)
def test_a_question_naming_emergency_symptoms_short_circuits(mock_llm):
    # Very often this is someone describing themselves, not asking trivia.
    response = _ask(
        {"patient_id": PATIENT_ID, "question": "what should I do about chest pain"}
    )

    body = response.json()
    assert body["needs_assessment"] is True
    mock_llm.assert_not_called()


@patch("services.llm_client.get_medical_answer", return_value=MOCK_ANSWER)
def test_answer_always_carries_a_disclaimer(mock_llm):
    body = _ask({"patient_id": PATIENT_ID, "question": "What is a CBC test?"}).json()

    assert "not a medical diagnosis" in body["answer"].lower()


@patch("services.llm_client.get_medical_answer", return_value=MOCK_ANSWER)
def test_arabic_question_requests_an_arabic_answer(mock_llm):
    _ask({"patient_id": PATIENT_ID, "question": "ما هو تحليل الدم الشامل؟"})

    assert mock_llm.call_args.kwargs["language"] == "ar"


def test_empty_question_is_rejected_gracefully():
    assert _ask({"patient_id": PATIENT_ID, "question": ""}).status_code == 422
