from typing import Literal, Optional

from pydantic import BaseModel, Field

Urgency = Literal["routine", "urgent", "emergency"]


class WaitingPatient(BaseModel):
    appointment_id: str
    scheduled_time: str = Field(..., description='24-hour "HH:MM"')
    urgency: Urgency = "routine"
    # How long they have already been waiting. Drives the fairness rule that
    # stops routine patients being overtaken indefinitely.
    waited_minutes: int = Field(0, ge=0, le=1440)
    patient_name: Optional[str] = None
    reason: Optional[str] = None


class QueuePriorityRequest(BaseModel):
    doctor_id: str
    patients: list[WaitingPatient] = Field(..., min_length=1, max_length=100)


class PrioritisedPatient(BaseModel):
    appointment_id: str
    patient_name: Optional[str] = None
    position: int = Field(..., ge=1)
    previous_position: int = Field(..., ge=1)
    moved: int = Field(..., description="Places gained (+) or lost (-)")
    urgency: Urgency
    priority_score: float
    reason: str


class QueuePriorityResponse(BaseModel):
    queue: list[PrioritisedPatient]
    changes_made: int
    explanation: str
