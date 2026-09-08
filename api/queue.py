from fastapi import APIRouter, Depends

from core.auth import verify_token
from schemas.queue import QueueRequest, QueueResponse
from services import queue_predictor

router = APIRouter()


@router.post("/predict-queue", response_model=QueueResponse)
def predict_queue(request: QueueRequest, _patient_sub: str = Depends(verify_token)) -> QueueResponse:
    result = queue_predictor.predict(request)
    return QueueResponse(**result)
