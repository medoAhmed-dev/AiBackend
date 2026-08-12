from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["patient", "ai"]
    content: str


class ChatRequest(BaseModel):
    patient_id: str
    message: str = Field(..., min_length=1)
    conversation_history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    urgency: Literal["green", "yellow", "red"]
    possible_causes: list[str]
    recommended_specialty: Optional[str] = None
    suggested_tests: list[str]
    follow_up_questions: list[str]
    emergency_flag: bool
