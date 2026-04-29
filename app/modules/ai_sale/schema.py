from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AISaleState(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    from_web: str = ""
    mode: str = "discovery"

    requirements: Dict[str, Any] = Field(default_factory=dict)
    missing_requirements: List[str] = Field(default_factory=list)
    requirement_ready: bool = False

    search_query: Optional[str] = None
    recommended_courses: List[dict] = Field(default_factory=list)

    last_user_message: Optional[str] = None
    last_answer: Optional[str] = None
    last_step: Optional[str] = None
    conversation_history: List[dict] = Field(default_factory=list)
    recommended_course_cta: List[dict] = Field(default_factory=list)

class AISaleRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=36)
    user_message: str = Field(..., min_length=1)

class AISaleResetRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=36)