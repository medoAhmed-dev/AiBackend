from unittest.mock import patch

from fastapi.testclient import TestClient

from core.auth import verify_token
from main import app

app.dependency_overrides[verify_token] = lambda: "test-patient-sub"

client = TestClient(app)

PATIENT_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

MOCK_MILD_RESPONSE = {
    "reply": "A runny nose and mild sore throat are usually a common cold.",
    "urgency": "green",
    "possible_causes": ["common cold", "seasonal allergies"],
    "recommended_specialty": None,
    "suggested_tests": [],
    "follow_up_questions": ["How long have you had these symptoms?"],
    "emergency_flag": False,
}

MOCK_MILD_RESPONSE_AR = {
    "reply": "يبدو أنك تعاني من نزلة برد خفيفة. ينصح بالراحة وشرب السوائل والمتابعة إذا ساءت الأعراض.",
    "urgency": "green",
    "possible_causes": ["نزلة برد", "حساسية موسمية"],
    "recommended_specialty": None,
    "suggested_tests": [],
    "follow_up_questions": ["منذ متى وأنت تعاني من هذه الأعراض؟"],
    "emergency_flag": False,
}


def _post_chat(payload: dict):
    return client.post("/chat", json=payload, headers={"Authorization": "Bearer test-token"})


@patch("services.llm_client.get_llm_response", return_value=MOCK_MILD_RESPONSE)
def test_mild_symptom_returns_non_emergency_urgency(mock_llm):
    response = _post_chat(
        {
            "patient_id": PATIENT_ID,
            "message": "I have a runny nose and a mild sore throat.",
            "conversation_history": [],
        }
    )

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in ("green", "yellow")
    assert body["emergency_flag"] is False


@patch("services.llm_client.get_llm_response", return_value=MOCK_MILD_RESPONSE)
def test_emergency_symptom_forces_red_regardless_of_model_output(mock_llm):
    response = _post_chat(
        {
            "patient_id": PATIENT_ID,
            "message": "I have crushing chest pain and I can't breathe.",
            "conversation_history": [],
        }
    )

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "red"
    assert body["emergency_flag"] is True
    # The hardcoded safety layer must short-circuit before the model is consulted.
    mock_llm.assert_not_called()


def test_empty_message_is_rejected_gracefully_not_a_crash():
    response = _post_chat(
        {
            "patient_id": PATIENT_ID,
            "message": "",
            "conversation_history": [],
        }
    )

    assert response.status_code == 422


@patch("services.llm_client.get_llm_response", return_value=MOCK_MILD_RESPONSE)
def test_multi_turn_conversation_forwards_history_to_llm(mock_llm):
    history = [
        {"role": "patient", "content": "I've had a headache since yesterday."},
        {"role": "ai", "content": "How severe is the pain on a scale of 1-10?"},
    ]

    response = _post_chat(
        {
            "patient_id": PATIENT_ID,
            "message": "It's about a 4, comes and goes.",
            "conversation_history": history,
        }
    )

    assert response.status_code == 200
    mock_llm.assert_called_once()
    called_message, called_history = mock_llm.call_args[0]
    assert called_message == "It's about a 4, comes and goes."
    assert len(called_history) == 2
    assert called_history[0].content == "I've had a headache since yesterday."


@patch("services.llm_client.get_llm_response", return_value=MOCK_MILD_RESPONSE_AR)
def test_arabic_emergency_symptom_forces_red_regardless_of_model_output(mock_llm):
    response = _post_chat(
        {
            "patient_id": PATIENT_ID,
            "message": "عندي ألم شديد في الصدر ومش قادر اتنفس",
            "conversation_history": [],
        }
    )

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "red"
    assert body["emergency_flag"] is True
    assert any("؀" <= ch <= "ۿ" for ch in body["reply"])
    # The hardcoded safety layer must short-circuit before the model is consulted.
    mock_llm.assert_not_called()


@patch("services.llm_client.get_llm_response", return_value=MOCK_MILD_RESPONSE_AR)
def test_mild_arabic_symptom_returns_coherent_arabic_reply(mock_llm):
    response = _post_chat(
        {
            "patient_id": PATIENT_ID,
            "message": "عندي رشح خفيف وصداع بسيط من يومين.",
            "conversation_history": [],
        }
    )

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in ("green", "yellow")
    assert body["emergency_flag"] is False
    # reply stays in Arabic, including the appended disclaimer
    assert any("؀" <= ch <= "ۿ" for ch in body["reply"])
    assert "الطبيب" in body["reply"]
