from pydantic import BaseModel, Field
from typing import Optional


class ChatState_aiselflearning(BaseModel):
    chat_id: Optional[str] = None
    OCourse_no: Optional[int] = None
    last_user_message: Optional[str] = None
    last_answer: Optional[str] = None


class ChatRequest_aiselflearning(BaseModel):
    chat_id: str
    OCourse_no: int
    user_message: str
    state: Optional[ChatState_aiselflearning] = None


class ChatResponse_aiselflearning(BaseModel):
    reply: str
    state: Optional[ChatState_aiselflearning] = None
    source: Optional[str] = None
    chat_id: str