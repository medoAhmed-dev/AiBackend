from fastapi.testclient import TestClient

from core.auth import verify_token
from main import app

app.dependency_overrides[verify_token] = lambda: "test-patient-sub"

client = TestClient(app)

BASE_PAYLOAD = {
    "doctor_id": "c3d4e5f6-a7b8-9012-cdef-345678901234",
    "current_queue_length": 5,
    "average_consultation_minutes": 12,
    "time_of_day": "14:30",
    "day_of_week": "Tuesday",
    "emergency_patients_ahead": 0,
}


def _post_queue(payload: dict):
    return client.post(
        "/predict-queue", json=payload, headers={"Authorization": "Bearer test-token"}
    )


def test_normal_queue_length_gives_sane_wait_time():
    response = _post_queue(BASE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["estimated_wait_minutes"] <= 300
    assert body["queue_position"] == BASE_PAYLOAD["current_queue_length"] + 1
    assert body["confidence"] in ("low", "medium", "high")
    assert body["explanation"]


def test_zero_queue_length_gives_near_minimal_wait():
    payload = {**BASE_PAYLOAD, "current_queue_length": 0, "emergency_patients_ahead": 0}
    response = _post_queue(payload)

    assert response.status_code == 200
    body = response.json()
    assert body["queue_position"] == 1
    # Near-minimal: well under a full queue's worth of consultations.
    assert body["estimated_wait_minutes"] < BASE_PAYLOAD["average_consultation_minutes"] * 3


def test_emergency_patients_ahead_increases_wait_time():
    no_emergency = _post_queue({**BASE_PAYLOAD, "emergency_patients_ahead": 0}).json()
    with_emergency = _post_queue({**BASE_PAYLOAD, "emergency_patients_ahead": 3}).json()

    assert with_emergency["estimated_wait_minutes"] > no_emergency["estimated_wait_minutes"]


def test_missing_field_is_rejected_gracefully_not_a_crash():
    payload = dict(BASE_PAYLOAD)
    del payload["current_queue_length"]

    response = _post_queue(payload)

    assert response.status_code == 422


def test_malformed_time_is_rejected_gracefully_not_a_crash():
    response = _post_queue({**BASE_PAYLOAD, "time_of_day": "not-a-time"})

    assert response.status_code == 422


def test_malformed_day_is_rejected_gracefully_not_a_crash():
    response = _post_queue({**BASE_PAYLOAD, "day_of_week": "Someday"})

    assert response.status_code == 422
