import logging

from fastapi import APIRouter, Depends, HTTPException

from core.auth import verify_token
from schemas.chat import ChatRequest, ChatResponse
from services import llm_client, safety

logger = logging.getLogger("medical_chat.api")

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _patient_sub: str = Depends(verify_token)) -> ChatResponse:
    safety.flag_out_of_scope(request.message, request.patient_id)

    if safety.detect_red_flag(request.message):
        return ChatResponse(
            reply=safety.emergency_reply_for(request.message),
            urgency="red",
            possible_causes=[],
            recommended_specialty=None,
            suggested_tests=[],
            follow_up_questions=[],
            emergency_flag=True,
        )

    try:
        raw = llm_client.get_llm_response(request.message, request.conversation_history)
        response = ChatResponse(**raw)
    except Exception:
        logger.exception("LLM call failed for patient_id=%s", request.patient_id)
        raise HTTPException(status_code=502, detail="AI service returned an unexpected response")

    if response.emergency_flag and response.urgency != "red":
        response.urgency = "red"

    response.reply = safety.ensure_disclaimer(response.reply)
    return response
