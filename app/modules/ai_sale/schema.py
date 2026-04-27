from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AISaleState(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None

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


class AISaleRequest(BaseModel):
    user_message: str
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    state: Optional[AISaleState] = None


class AISaleResetRequest(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None