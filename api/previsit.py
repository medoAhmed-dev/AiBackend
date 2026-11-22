import logging

from fastapi import APIRouter, Depends, HTTPException

from core.auth import verify_token
from schemas.previsit import (
    MedicalQuestionRequest,
    MedicalQuestionResponse,
    PreVisitSummaryRequest,
    PreVisitSummaryResponse,
)
from services import llm_client, previsit_intake, safety

logger = logging.getLogger("medical_chat.api")

router = APIRouter()


@router.post("/previsit-summary", response_model=PreVisitSummaryResponse)
def previsit_summary(
    request: PreVisitSummaryRequest, _patient_sub: str = Depends(verify_token)
) -> PreVisitSummaryResponse:
    scannable = previsit_intake.scannable_text(request)
    safety.flag_out_of_scope(scannable, request.patient_id)

    try:
        raw = llm_client.get_previsit_summary(
            previsit_intake.render(request),
            language=safety.detect_language(scannable),
        )
        response = PreVisitSummaryResponse(**raw)
    except Exception:
        logger.exception("Pre-visit summary failed for patient_id=%s", request.patient_id)
        raise HTTPException(status_code=502, detail="AI service returned an unexpected response")

    # A red flag in what the patient wrote must reach the clinician even if
    # the model didn't surface it. Same principle as the chat safety layer:
    # this doesn't depend on the model having a good day.
    matched = safety.matched_red_flags(scannable)
    if matched:
        existing = " ".join(response.flags_for_clinician).lower()
        for flag in matched:
            if flag.lower() not in existing:
                response.flags_for_clinician.insert(0, f"Patient reported: {flag}")

    return response


@router.post("/ask", response_model=MedicalQuestionResponse)
def ask(
    request: MedicalQuestionRequest, _patient_sub: str = Depends(verify_token)
) -> MedicalQuestionResponse:
    safety.flag_out_of_scope(request.question, request.patient_id)

    # A general question that names emergency symptoms is very often someone
    # describing their own situation. Answering it as trivia would be wrong,
    # so short-circuit exactly as /chat does.
    if safety.detect_red_flag(request.question):
        return MedicalQuestionResponse(
            answer=safety.emergency_reply_for(request.question),
            key_points=[],
            needs_assessment=True,
            related_questions=[],
        )

    try:
        raw = llm_client.get_medical_answer(
            request.question,
            language=safety.detect_language(request.question),
        )
        response = MedicalQuestionResponse(**raw)
    except Exception:
        logger.exception("Medical Q&A failed for patient_id=%s", request.patient_id)
        raise HTTPException(status_code=502, detail="AI service returned an unexpected response")

    response.answer = safety.ensure_disclaimer(response.answer)
    return response
