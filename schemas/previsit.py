from typing import Literal, Optional

from pydantic import BaseModel, Field


class PreVisitSummaryRequest(BaseModel):
    patient_id: str
    reason_for_visit: str = Field(..., min_length=1, max_length=1000)
    symptoms: list[str] = Field(default_factory=list, max_length=20)
    duration_days: Optional[float] = Field(None, ge=0, le=3650)
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[Literal["male", "female", "other"]] = None
    # Passed in by the caller from the patient record — this endpoint is
    # stateless and reads nothing from the database itself.
    medical_conditions: Optional[str] = Field(None, max_length=2000)
    allergies: Optional[str] = Field(None, max_length=1000)
    current_medications: Optional[str] = Field(None, max_length=1000)
    additional_notes: Optional[str] = Field(None, max_length=2000)


class PreVisitSummaryResponse(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)
    questions_for_doctor: list[str] = Field(default_factory=list)
    flags_for_clinician: list[str] = Field(default_factory=list)


class MedicalQuestionRequest(BaseModel):
    patient_id: str
    question: str = Field(..., min_length=1, max_length=2000)


class MedicalQuestionResponse(BaseModel):
    answer: str
    key_points: list[str] = Field(default_factory=list)
    # True when the question is really about the asker's own current
    # symptoms — the app should steer them to the symptom checker instead
    # of treating a general answer as sufficient.
    needs_assessment: bool = False
    related_questions: list[str] = Field(default_factory=list)
