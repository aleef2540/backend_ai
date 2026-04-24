from pydantic import BaseModel, Field
from typing import Optional, List


class ChatState_aicustom(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    course_use: List[int] = Field(default_factory=list)

    mode: str = "idle"
    intent: str = "unknown"

    topic: str = "unknown"
    active_course_no: Optional[int] = None

    last_intent: str = "unknown"
    last_answer_type: Optional[str] = None
    last_user_message: Optional[str] = None
    last_answer: Optional[str] = None


class ChatRequest_aicustom(BaseModel):
    user_message: str
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    course_use: List[str] = Field(default_factory=list)
    state: Optional[ChatState_aicustom] = None


class ChatResponse_aicustom(BaseModel):
    reply: str
    state: Optional[ChatState_aicustom] = None
    source: Optional[str] = None
    active_video: Optional[dict] = None


class ResetRequest_aicustom(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None