from typing import Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["mild", "moderate", "severe"]
Urgency = Literal["green", "yellow", "red"]
Likelihood = Literal["low", "moderate", "high"]
RecommendedAction = Literal["self_care", "see_doctor_soon", "urgent_care", "emergency"]


class SymptomCheckRequest(BaseModel):
    patient_id: str
    symptoms: list[str] = Field(..., min_length=1, max_length=20)
    duration_days: Optional[float] = Field(None, ge=0, le=3650)
    severity: Optional[Severity] = None
    # Age and sex change what's likely for the same symptoms. Passed in by the
    # caller rather than read from the database — this endpoint is stateless,
    # same as /chat.
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[Literal["male", "female", "other"]] = None
    additional_notes: Optional[str] = Field(None, max_length=2000)


class PossibleCondition(BaseModel):
    name: str
    likelihood: Likelihood
    reasoning: str


class SymptomCheckResponse(BaseModel):
    assessment: str
    urgency: Urgency
    possible_conditions: list[PossibleCondition]
    recommended_action: RecommendedAction
    recommended_specialty: Optional[str] = None
    red_flags: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    emergency_flag: bool
