"""
Re-orders a doctor's live queue when urgency differs from booking order.

Rule-based and fully explainable, deliberately. Two reasons: there is no
historical "how should this queue have been ordered" data to learn from,
and more importantly a patient told they have been moved down the queue
deserves a reason that can be stated plainly. An unexplainable score is
not acceptable for this.

The one subtlety worth knowing is the fairness rule. Ordering purely by
urgency starves routine patients: on a busy day a steady trickle of
urgent cases could leave someone waiting indefinitely, which is both
unfair and clinically unwise. So waiting time escalates priority, and
past a threshold a long-waiting routine patient outranks a freshly
arrived urgent one.
"""

from schemas.queue_priority import (
    PrioritisedPatient,
    QueuePriorityRequest,
    WaitingPatient,
)

_URGENCY_BASE = {
    "emergency": 1000.0,
    "urgent": 100.0,
    "routine": 10.0,
}

# Each minute waited adds this much priority. Tuned so a routine patient
# overtakes a just-arrived urgent one after roughly 45 minutes, rather than
# waiting behind every urgent case that arrives.
_WAIT_ESCALATION_PER_MINUTE = 2.0

# The escalation is capped, and the cap matters for safety: uncapped, a
# routine patient waiting long enough would eventually outscore an emergency
# (at 2.0/min, 24 hours of waiting reaches 2890 against an emergency's 1000).
# A test caught exactly that. The cap sits above the routine-to-urgent gap of
# 90, so the fairness rule still works, but far below the 900 gap up to
# emergency, so no amount of waiting can ever overtake one.
_MAX_WAIT_BONUS = 300.0

_LONG_WAIT_MINUTES = 45


def _time_to_minutes(value: str) -> int:
    try:
        hours, minutes = value.split(":")
        return int(hours) * 60 + int(minutes)
    except (ValueError, AttributeError):
        return 0


def _score(patient: WaitingPatient) -> float:
    base = _URGENCY_BASE.get(patient.urgency, _URGENCY_BASE["routine"])
    wait_bonus = min(
        patient.waited_minutes * _WAIT_ESCALATION_PER_MINUTE, _MAX_WAIT_BONUS
    )
    return base + wait_bonus


def _reason(patient: WaitingPatient, moved: int) -> str:
    if patient.urgency == "emergency":
        return "Emergency case — seen first"
    if moved > 0 and patient.urgency == "urgent":
        return f"Urgent case — moved up {moved} place(s)"
    if moved > 0 and patient.waited_minutes >= _LONG_WAIT_MINUTES:
        return f"Waiting {patient.waited_minutes} minutes — moved up {moved} place(s)"
    if moved < 0:
        return f"Moved back {abs(moved)} place(s) for a more urgent case"
    if patient.urgency == "urgent":
        return "Urgent case"
    return "Scheduled order"


def prioritise(request: QueuePriorityRequest) -> dict:
    # Booking order is the baseline: whoever was scheduled earliest is first
    # before any urgency is considered.
    by_schedule = sorted(
        request.patients, key=lambda p: (_time_to_minutes(p.scheduled_time), p.appointment_id)
    )
    original_position = {p.appointment_id: i + 1 for i, p in enumerate(by_schedule)}

    # Ties fall back to scheduled order so the result is deterministic.
    by_priority = sorted(
        by_schedule,
        key=lambda p: (-_score(p), _time_to_minutes(p.scheduled_time), p.appointment_id),
    )

    queue: list[PrioritisedPatient] = []
    changes = 0
    for index, patient in enumerate(by_priority):
        position = index + 1
        previous = original_position[patient.appointment_id]
        moved = previous - position
        if moved != 0:
            changes += 1
        queue.append(
            PrioritisedPatient(
                appointment_id=patient.appointment_id,
                patient_name=patient.patient_name,
                position=position,
                previous_position=previous,
                moved=moved,
                urgency=patient.urgency,
                priority_score=round(_score(patient), 1),
                reason=_reason(patient, moved),
            )
        )

    if changes == 0:
        explanation = "No reordering needed — booking order already matches clinical priority."
    else:
        emergencies = sum(1 for p in request.patients if p.urgency == "emergency")
        parts = [f"{changes} patient(s) changed position"]
        if emergencies:
            parts.append(f"{emergencies} emergency case(s) moved to the front")
        parts.append(
            f"patients waiting over {_LONG_WAIT_MINUTES} minutes are escalated so they "
            "are not overtaken indefinitely"
        )
        explanation = "; ".join(parts) + "."

    return {"queue": queue, "changes_made": changes, "explanation": explanation}
