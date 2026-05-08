from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: str
    
class StepAnswer(BaseModel):
    fixed_question: str = ""
    asked_question: str = ""
    user_answer: str = ""
    status: str = ""
    probe_count: int = 0
    is_completed: bool = False
    summary: str = ""

class ChatState(BaseModel):
    phase: int = 1
    step: int = 0
    retry_count: int = 0
    last_question: str = ""
    # คำตอบราย step
    answers_by_step: Dict[int, StepAnswer] = Field(default_factory=dict)
    # คำตอบแบบ semantic
    answers: Dict[str, Any] = Field(default_factory=dict)
    # ประวัติสนทนา
    history: List[Dict[str, Any]] = Field(default_factory=list)

class ChatRequest_aicoach(BaseModel):
    user_message: str = Field(..., min_length=1)
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    state: Optional[ChatState] = None

class ChatResponse_aicoach(BaseModel):
    reply: str
    state: Optional[ChatState] = None
    source: Optional[str] = None

class ResetRequest_aicaoch(BaseModel):
    web_no: int
    member_no: int
