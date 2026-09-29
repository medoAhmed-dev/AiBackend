from unittest.mock import patch

from fastapi.testclient import TestClient

from core.auth import verify_token
from main import app

app.dependency_overrides[verify_token] = lambda: "test-patient-sub"

client = TestClient(app)

PATIENT_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

BASE_PAYLOAD = {
    "patient_id": PATIENT_ID,
    "symptoms": ["runny nose", "mild sore throat"],
    "duration_days": 2,
    "severity": "mild",
    "age": 25,
    "sex": "male",
}

MOCK_MILD = {
    "assessment": "This looks like a common cold and can usually be managed at home.",
    "urgency": "green",
    "possible_conditions": [
        {
            "name": "Common cold",
            "likelihood": "high",
            "reasoning": "Runny nose with a mild sore throat over two days is typical.",
        }
    ],
    "recommended_action": "self_care",
    "recommended_specialty": None,
    "red_flags": [],
    "follow_up_questions": ["Do you have a fever?"],
    "emergency_flag": False,
}

MOCK_MILD_AR = {
    "assessment": "يبدو أن الأعراض ناتجة عن نزلة برد خفيفة ويمكن متابعتها في المنزل.",
    "urgency": "green",
    "possible_conditions": [
        {
            "name": "نزلة برد",
            "likelihood": "high",
            "reasoning": "رشح مع التهاب بسيط في الحلق لمدة يومين.",
        }
    ],
    "recommended_action": "self_care",
    "recommended_specialty": None,
    "red_flags": [],
    "follow_up_questions": ["هل لديك ارتفاع في درجة الحرارة؟"],
    "emergency_flag": False,
}


def _post(payload: dict):
    return client.post(
        "/symptom-check", json=payload, headers={"Authorization": "Bearer test-token"}
    )


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD)
def test_mild_symptoms_return_self_care(mock_llm):
    response = _post(BASE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "green"
    assert body["recommended_action"] == "self_care"
    assert body["emergency_flag"] is False
    assert body["possible_conditions"][0]["name"] == "Common cold"


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD)
def test_red_flag_symptom_forces_emergency_without_calling_the_model(mock_llm):
    response = _post({**BASE_PAYLOAD, "symptoms": ["chest pain", "shortness of breath"]})

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "red"
    assert body["recommended_action"] == "emergency"
    assert body["emergency_flag"] is True
    assert "chest pain" in body["red_flags"]
    mock_llm.assert_not_called()


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD)
def test_red_flag_in_free_text_notes_is_caught_too(mock_llm):
    response = _post(
        {
            **BASE_PAYLOAD,
            "symptoms": ["tired"],
            "additional_notes": "Also I think I am having a stroke, my face is drooping.",
        }
    )

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "red"
    assert body["emergency_flag"] is True
    mock_llm.assert_not_called()


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD)
def test_arabic_red_flag_forces_emergency(mock_llm):
    response = _post({**BASE_PAYLOAD, "symptoms": ["ألم في الصدر", "دوخة"]})

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "red"
    assert body["emergency_flag"] is True
    # Emergency guidance comes back in the language the patient used.
    assert any("؀" <= ch <= "ۿ" for ch in body["assessment"])
    mock_llm.assert_not_called()


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD_AR)
def test_arabic_report_returns_arabic_assessment(mock_llm):
    response = _post({**BASE_PAYLOAD, "symptoms": ["رشح", "صداع خفيف"]})

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in ("green", "yellow")
    assert any("؀" <= ch <= "ۿ" for ch in body["assessment"])
    assert "الطبيب" in body["assessment"]


@patch("services.llm_client.get_symptom_assessment")
def test_model_emergency_flag_forces_red_and_emergency_action(mock_llm):
    # Model contradicts itself: flags an emergency but downgrades the rest.
    mock_llm.return_value = {
        **MOCK_MILD,
        "urgency": "yellow",
        "recommended_action": "see_doctor_soon",
        "emergency_flag": True,
    }

    body = _post(BASE_PAYLOAD).json()

    assert body["urgency"] == "red"
    assert body["recommended_action"] == "emergency"


def test_empty_symptom_list_is_rejected_gracefully():
    response = _post({**BASE_PAYLOAD, "symptoms": []})

    assert response.status_code == 422


def test_invalid_severity_is_rejected_gracefully():
    response = _post({**BASE_PAYLOAD, "severity": "catastrophic"})

    assert response.status_code == 422


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD)
def test_optional_fields_can_be_omitted(mock_llm):
    response = _post({"patient_id": PATIENT_ID, "symptoms": ["headache"]})

    assert response.status_code == 200
    intake = mock_llm.call_args[0][0]
    # Omitted fields shouldn't be rendered into the intake at all.
    assert "Severity" not in intake
    assert "Age" not in intake


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD)
def test_english_report_instructs_the_model_to_answer_in_english(mock_llm):
    # Regression: asked to infer the language, the model answered English
    # reports in Arabic every time. The language is now stated explicitly.
    _post({**BASE_PAYLOAD, "symptoms": ["sore throat", "fever"]})

    intake = mock_llm.call_args[0][0]
    assert "Write your entire response in English." in intake


@patch("services.llm_client.get_symptom_assessment", return_value=MOCK_MILD_AR)
def test_arabic_report_instructs_the_model_to_answer_in_arabic(mock_llm):
    _post({**BASE_PAYLOAD, "symptoms": ["رشح", "صداع خفيف"]})

    intake = mock_llm.call_args[0][0]
    assert "اكتب ردك بالكامل باللغة العربية." in intake
