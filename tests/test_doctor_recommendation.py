from fastapi.testclient import TestClient

from core.auth import verify_token
from main import app

app.dependency_overrides[verify_token] = lambda: "test-patient-sub"

client = TestClient(app)

PATIENT_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# Only the fields the real `doctors` table actually has today.
SCHEMA_ONLY_DOCTORS = [
    {
        "id": "d1",
        "full_name": "Dr. Amira Hassan",
        "specialty": "Cardiology",
        "years_of_experience": 18,
        "consultation_fee": 400,
        "is_verified": True,
        "hospital_name": "Nile Heart Center",
    },
    {
        "id": "d2",
        "full_name": "Dr. Omar Fathy",
        "specialty": "Cardiology",
        "years_of_experience": 3,
        "consultation_fee": 250,
        "is_verified": True,
        "hospital_name": "Cairo General",
    },
    {
        "id": "d3",
        "full_name": "Dr. Laila Mansour",
        "specialty": "Dermatology",
        "years_of_experience": 12,
        "consultation_fee": 300,
        "is_verified": True,
        "hospital_name": "Cairo General",
    },
]


def _post(payload: dict):
    return client.post(
        "/recommend-doctor", json=payload, headers={"Authorization": "Bearer test-token"}
    )


def test_ranks_doctors_and_reports_which_signals_it_could_use():
    body = _post({"patient_id": PATIENT_ID, "doctors": SCHEMA_ONLY_DOCTORS}).json()

    assert len(body["recommendations"]) == 3
    assert body["total_considered"] == 3
    # Works on today's schema alone, and says what it was missing.
    assert "experience" in body["signals_used"]
    assert "rating" in body["signals_unavailable"]
    assert "distance" in body["signals_unavailable"]


def test_specialty_mismatch_is_excluded_entirely():
    body = _post(
        {
            "patient_id": PATIENT_ID,
            "needed_specialty": "Cardiology",
            "doctors": SCHEMA_ONLY_DOCTORS,
        }
    ).json()

    returned = {r["doctor_id"] for r in body["recommendations"]}
    assert returned == {"d1", "d2"}  # the dermatologist is gone
    assert body["excluded_count"] == 1


def test_more_experience_outranks_less_when_all_else_is_equal():
    body = _post(
        {
            "patient_id": PATIENT_ID,
            "needed_specialty": "Cardiology",
            "doctors": SCHEMA_ONLY_DOCTORS,
        }
    ).json()

    top = body["recommendations"][0]
    assert top["doctor_id"] == "d1"  # 18 years beats 3
    assert any("18 years" in reason for reason in top["reasons"])


def test_optional_future_fields_are_used_when_present():
    doctors = [
        {**SCHEMA_ONLY_DOCTORS[1], "rating": 4.9, "distance_km": 1.0, "available_today": True},
        {**SCHEMA_ONLY_DOCTORS[0], "rating": 2.1, "distance_km": 40.0, "available_today": False},
    ]

    body = _post(
        {"patient_id": PATIENT_ID, "needed_specialty": "Cardiology", "doctors": doctors}
    ).json()

    # The less experienced doctor now wins on rating, distance and availability.
    assert body["recommendations"][0]["doctor_id"] == "d2"
    assert "rating" in body["signals_used"]
    assert body["signals_unavailable"] == []


def test_budget_filter_excludes_doctors_over_the_limit():
    body = _post(
        {
            "patient_id": PATIENT_ID,
            "needed_specialty": "Cardiology",
            "max_consultation_fee": 300,
            "doctors": SCHEMA_ONLY_DOCTORS,
        }
    ).json()

    returned = {r["doctor_id"] for r in body["recommendations"]}
    assert returned == {"d2"}  # d1 costs 400


def test_require_verified_excludes_unverified_doctors():
    doctors = [{**SCHEMA_ONLY_DOCTORS[0], "is_verified": False}, SCHEMA_ONLY_DOCTORS[1]]

    body = _post(
        {"patient_id": PATIENT_ID, "require_verified": True, "doctors": doctors}
    ).json()

    assert {r["doctor_id"] for r in body["recommendations"]} == {"d2"}


def test_limit_caps_how_many_come_back():
    body = _post({"patient_id": PATIENT_ID, "doctors": SCHEMA_ONLY_DOCTORS, "limit": 1}).json()

    assert len(body["recommendations"]) == 1


def test_every_recommendation_explains_itself():
    body = _post({"patient_id": PATIENT_ID, "doctors": SCHEMA_ONLY_DOCTORS}).json()

    for rec in body["recommendations"]:
        assert rec["reasons"], "a recommendation with no reasons can't be shown to a patient"
        assert 0 <= rec["score"] <= 100


def test_empty_doctor_list_is_rejected_gracefully():
    assert _post({"patient_id": PATIENT_ID, "doctors": []}).status_code == 422


def test_filters_can_exclude_everyone_without_crashing():
    body = _post(
        {
            "patient_id": PATIENT_ID,
            "needed_specialty": "Neurosurgery",
            "doctors": SCHEMA_ONLY_DOCTORS,
        }
    ).json()

    assert body["recommendations"] == []
    assert body["excluded_count"] == 3
