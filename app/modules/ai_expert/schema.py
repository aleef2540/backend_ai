from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Union


class ChatState_aiexpert(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    course_use: List[Union[int, str]] = Field(default_factory=list)

    mode: str = "idle"
    intent: str = "unknown"

    last_intent: str = "unknown"
    last_answer_type: Optional[str] = None
    last_user_message: Optional[str] = None
    last_answer: Optional[str] = None

    conversation_history: list = Field(default_factory=list)

class ChatRequest_aiexpert(BaseModel):
    room_id: Optional[int] = None


    web_no: Optional[int] = None
    member_no: Optional[int] = None
    user_message: str

    course_use: List[Union[int, str]] = Field(default_factory=list)

    # PHP จะส่ง state_json จาก DB เข้ามา
    state: Optional[ChatState_aiexpert] = None

class ChatResponse_aiexpert(BaseModel):
    room_id: Optional[int] = None
    reply: str
    state: Optional[ChatState_aiexpert] = None
    source: Optional[str] = None
    status: Optional[str] = None
    reason: Optional[str] = None
    active_video: Optional[dict] = None

class ResetRequest_aiexpert(BaseModel):
    room_id: int
    web_no: Optional[int] = None
    member_no: Optional[int] = None