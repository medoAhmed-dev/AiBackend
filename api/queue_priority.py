from fastapi import APIRouter, Depends

from core.auth import verify_token
from schemas.queue_priority import QueuePriorityRequest, QueuePriorityResponse
from services import queue_prioritizer

router = APIRouter()


@router.post("/prioritize-queue", response_model=QueuePriorityResponse)
def prioritize_queue(
    request: QueuePriorityRequest, _sub: str = Depends(verify_token)
) -> QueuePriorityResponse:
    return QueuePriorityResponse(**queue_prioritizer.prioritise(request))
