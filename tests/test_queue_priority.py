from fastapi.testclient import TestClient

from core.auth import verify_token
from main import app

app.dependency_overrides[verify_token] = lambda: "test-sub"

client = TestClient(app)

DOCTOR_ID = "c3d4e5f6-a7b8-9012-cdef-345678901234"


def _post(patients: list[dict]):
    return client.post(
        "/prioritize-queue",
        json={"doctor_id": DOCTOR_ID, "patients": patients},
        headers={"Authorization": "Bearer test-token"},
    )


def _order(body: dict) -> list[str]:
    return [p["appointment_id"] for p in body["queue"]]


def test_booking_order_is_kept_when_everyone_is_routine():
    body = _post(
        [
            {"appointment_id": "c", "scheduled_time": "11:00"},
            {"appointment_id": "a", "scheduled_time": "09:00"},
            {"appointment_id": "b", "scheduled_time": "10:00"},
        ]
    ).json()

    assert _order(body) == ["a", "b", "c"]
    assert body["changes_made"] == 0


def test_emergency_goes_to_the_front():
    body = _post(
        [
            {"appointment_id": "a", "scheduled_time": "09:00"},
            {"appointment_id": "b", "scheduled_time": "10:00"},
            {"appointment_id": "emergency", "scheduled_time": "11:30", "urgency": "emergency"},
        ]
    ).json()

    assert _order(body)[0] == "emergency"
    assert body["changes_made"] > 0


def test_urgent_outranks_routine_but_not_emergency():
    body = _post(
        [
            {"appointment_id": "routine", "scheduled_time": "09:00"},
            {"appointment_id": "urgent", "scheduled_time": "10:00", "urgency": "urgent"},
            {"appointment_id": "emergency", "scheduled_time": "11:00", "urgency": "emergency"},
        ]
    ).json()

    assert _order(body) == ["emergency", "urgent", "routine"]


def test_a_long_wait_escalates_a_routine_patient_above_a_fresh_urgent_one():
    # Without this, a steady trickle of urgent cases could leave a routine
    # patient waiting indefinitely.
    body = _post(
        [
            {
                "appointment_id": "waited_ages",
                "scheduled_time": "09:00",
                "urgency": "routine",
                "waited_minutes": 120,
            },
            {
                "appointment_id": "just_arrived_urgent",
                "scheduled_time": "11:00",
                "urgency": "urgent",
                "waited_minutes": 0,
            },
        ]
    ).json()

    assert _order(body)[0] == "waited_ages"


def test_waiting_never_lets_a_routine_patient_overtake_an_emergency():
    body = _post(
        [
            {
                "appointment_id": "waited_all_day",
                "scheduled_time": "08:00",
                "urgency": "routine",
                "waited_minutes": 1440,
            },
            {
                "appointment_id": "emergency",
                "scheduled_time": "17:00",
                "urgency": "emergency",
                "waited_minutes": 0,
            },
        ]
    ).json()

    assert _order(body)[0] == "emergency"


def test_every_patient_gets_a_reason_and_a_position_delta():
    body = _post(
        [
            {"appointment_id": "a", "scheduled_time": "09:00"},
            {"appointment_id": "e", "scheduled_time": "10:00", "urgency": "emergency"},
        ]
    ).json()

    for entry in body["queue"]:
        assert entry["reason"], "a patient moved down the queue deserves a stated reason"
        assert entry["moved"] == entry["previous_position"] - entry["position"]


def test_ordering_is_deterministic_for_identical_input():
    patients = [
        {"appointment_id": "a", "scheduled_time": "09:00", "urgency": "urgent"},
        {"appointment_id": "b", "scheduled_time": "09:00", "urgency": "urgent"},
    ]
    assert _order(_post(patients).json()) == _order(_post(patients).json())


def test_empty_queue_is_rejected_gracefully():
    assert _post([]).status_code == 422


def test_invalid_urgency_is_rejected_gracefully():
    response = _post(
        [{"appointment_id": "a", "scheduled_time": "09:00", "urgency": "catastrophic"}]
    )
    assert response.status_code == 422
