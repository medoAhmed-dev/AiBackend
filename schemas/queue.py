import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Day = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class QueueRequest(BaseModel):
    doctor_id: str
    current_queue_length: int = Field(..., ge=0)
    average_consultation_minutes: float = Field(..., gt=0)
    time_of_day: str
    day_of_week: Day
    emergency_patients_ahead: int = Field(0, ge=0)

    @field_validator("time_of_day")
    @classmethod
    def validate_time_of_day(cls, value: str) -> str:
        if not _TIME_RE.match(value):
            raise ValueError('time_of_day must be in 24-hour "HH:MM" format')
        return value


class QueueResponse(BaseModel):
    estimated_wait_minutes: int = Field(..., ge=0)
    queue_position: int = Field(..., ge=1)
    confidence: Literal["low", "medium", "high"]
    explanation: str
