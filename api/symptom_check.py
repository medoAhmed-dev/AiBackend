import logging

from fastapi import APIRouter, Depends, HTTPException

from core.auth import verify_token
from schemas.symptom_check import SymptomCheckRequest, SymptomCheckResponse
from services import llm_client, safety, symptom_intake

logger = logging.getLogger("medical_chat.api")

router = APIRouter()


@router.post("/symptom-check", response_model=SymptomCheckResponse)
def symptom_check(
    request: SymptomCheckRequest, _patient_sub: str = Depends(verify_token)
) -> SymptomCheckResponse:
    scannable = symptom_intake.scannable_text(request)
    safety.flag_out_of_scope(scannable, request.patient_id)

    # Same principle as /chat: a reported emergency never depends on the
    # model behaving well. Matched here, before the model is consulted.
    red_flags = safety.matched_red_flags(scannable)
    if red_flags:
        return SymptomCheckResponse(
            assessment=safety.emergency_reply_for(scannable),
            urgency="red",
            possible_conditions=[],
            recommended_action="emergency",
            recommended_specialty=None,
            red_flags=red_flags,
            follow_up_questions=[],
            emergency_flag=True,
        )

    try:
        raw = llm_client.get_symptom_assessment(
            symptom_intake.render(request),
            language=safety.detect_language(scannable),
        )
        response = SymptomCheckResponse(**raw)
    except Exception:
        logger.exception("Symptom assessment failed for patient_id=%s", request.patient_id)
        raise HTTPException(status_code=502, detail="AI service returned an unexpected response")

    # Urgency and action have to agree; the model is told this, but the
    # contract shouldn't depend on it complying.
    if response.emergency_flag or response.recommended_action == "emergency":
        response.emergency_flag = True
        response.urgency = "red"
        response.recommended_action = "emergency"

    response.assessment = safety.ensure_disclaimer(response.assessment)
    return response
