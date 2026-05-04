from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Union


class ChatState_aicustom(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    course_use: List[Union[int, str]] = Field(default_factory=list)

    mode: str = "idle"
    intent: str = "unknown"

    topic: str = "unknown"
    active_course_no: Optional[int] = None

    journey_name: Optional[str] = None

    last_intent: str = "unknown"
    last_answer_type: Optional[str] = None
    last_user_message: Optional[str] = None
    last_answer: Optional[str] = None

    requirements: dict = Field(default_factory=dict)
    missing_requirements: list = Field(default_factory=list)
    requirement_ready: bool = False
    conversation_history: list = Field(default_factory=list)
    search_query: Optional[str] = None
    matched_rag_results: list = Field(default_factory=list)

    allowed_course_data: list = Field(default_factory=list)
    allowed_course_name_context: Optional[str] = None

        # Learning / Feedback phase
    learning_phase: dict = Field(default_factory=dict)
    feedback_status: Optional[str] = None

    @field_validator("requirements", "learning_phase", mode="before")
    @classmethod
    def normalize_dict_fields(cls, v):
        if v in [None, "", []]:
            return {}
        return v



class ChatRequest_aicustom(BaseModel):
    room_id: Optional[int] = None


    web_no: Optional[int] = None
    member_no: Optional[int] = None
    user_message: str

    course_use: List[Union[int, str]] = Field(default_factory=list)

    # PHP จะส่ง state_json จาก DB เข้ามา
    state: Optional[ChatState_aicustom] = None


class ChatResponse_aicustom(BaseModel):
    room_id: Optional[int] = None
    reply: str
    state: Optional[ChatState_aicustom] = None
    source: Optional[str] = None
    status: Optional[str] = None
    reason: Optional[str] = None
    active_video: Optional[dict] = None


class ResetRequest_aicustom(BaseModel):
    room_id: int
    web_no: Optional[int] = None
    member_no: Optional[int] = None