from fastapi import APIRouter, Depends

from core.auth import verify_token
from schemas.doctor_recommendation import (
    DoctorRecommendationRequest,
    DoctorRecommendationResponse,
)
from services import doctor_scorer

router = APIRouter()


@router.post("/recommend-doctor", response_model=DoctorRecommendationResponse)
def recommend_doctor(
    request: DoctorRecommendationRequest, _patient_sub: str = Depends(verify_token)
) -> DoctorRecommendationResponse:
    return DoctorRecommendationResponse(**doctor_scorer.recommend(request))
