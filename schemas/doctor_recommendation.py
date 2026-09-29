from typing import Literal, Optional

from pydantic import BaseModel, Field


class DoctorCandidate(BaseModel):
    """Mirrors the real `doctors` table, plus three optional fields the
    schema doesn't have yet (rating, distance_km, available_today). Those are
    optional on purpose: the endpoint ranks correctly without them today, and
    starts using them automatically once the app team adds them — no change
    needed here."""

    id: str
    full_name: str
    specialty: str
    years_of_experience: Optional[int] = Field(None, ge=0, le=70)
    consultation_fee: Optional[float] = Field(None, ge=0)
    is_verified: Optional[bool] = None
    hospital_name: Optional[str] = None

    # Not yet in the database schema.
    rating: Optional[float] = Field(None, ge=0, le=5)
    distance_km: Optional[float] = Field(None, ge=0)
    available_today: Optional[bool] = None


class DoctorRecommendationRequest(BaseModel):
    patient_id: str
    needed_specialty: Optional[str] = None
    max_consultation_fee: Optional[float] = Field(None, ge=0)
    require_verified: bool = False
    doctors: list[DoctorCandidate] = Field(..., min_length=1, max_length=200)
    limit: int = Field(5, ge=1, le=50)


class ScoredDoctor(BaseModel):
    doctor_id: str
    full_name: str
    specialty: str
    score: float = Field(..., ge=0, le=100)
    reasons: list[str]
    consultation_fee: Optional[float] = None
    hospital_name: Optional[str] = None


class DoctorRecommendationResponse(BaseModel):
    recommendations: list[ScoredDoctor]
    total_considered: int
    excluded_count: int
    # Names which signals were actually available, so a caller (or a report)
    # can see what the ranking was and wasn't able to take into account.
    signals_used: list[str]
    signals_unavailable: list[str]
    method: Literal["rule_based_weighted"] = "rule_based_weighted"
