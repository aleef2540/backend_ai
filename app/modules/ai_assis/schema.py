from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AIAssisState(BaseModel):

    web_no: Optional[int] = None
    member_no: Optional[int] = None

    from_web: str = ""

    mode: str = "normal"

    current_intent: Optional[str] = None
    previous_intent: Optional[str] = None

    topic: Optional[str] = None

    # =========================
    # COURSE
    # =========================
    course_context: Dict[str, Any] = Field(default_factory=dict)

    recommended_courses: List[dict] = Field(default_factory=list)
    recommended_course_cta: List[dict] = Field(default_factory=list)

    matched_course: Optional[str] = None
    matched_course_id: Optional[str] = None

    # =========================
    # INSTRUCTOR
    # =========================
    instructor_context: Dict[str, Any] = Field(default_factory=dict)

    recommended_instructors: List[dict] = Field(default_factory=list)

    # =========================
    # COMPANY
    # =========================
    company_context: Dict[str, Any] = Field(default_factory=dict)

    # =========================
    # QUOTATION
    # =========================
    quotation_context: Dict[str, Any] = Field(default_factory=dict)

    pending_action: Optional[str] = None

    # =========================
    # CHAT
    # =========================
    last_user_message: Optional[str] = None
    last_answer: Optional[str] = None
    last_step: Optional[str] = None

    conversation_history: List[dict] = Field(default_factory=list)

class AISaleRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=36)
    user_message: str = Field(..., min_length=1)
    from_web: str = Field(..., min_length=1)

class AISaleResetRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=36)