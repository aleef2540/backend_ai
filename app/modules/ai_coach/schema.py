from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class StepAnswer(BaseModel):
    phase: int = 1
    step: int = 0
    rule_key: str = ""
    question: str = ""
    user_answer: str = ""
    status: str = ""
    is_completed: bool = False
    summary: str = ""
    extracted: Dict[str, Any] = Field(default_factory=dict)


class CoachingMemory(BaseModel):
    themes: List[str] = Field(default_factory=list)
    emotions: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    readiness: str = "unknown"  # unknown | low | medium | high
    scope_status: str = "unknown"  # unknown | valid | invalid | needs_reframe
    last_signal: str = ""
    last_policy_action: str = ""


class ChatState(BaseModel):
    phase: int = 1
    step: int = 0
    retry_count: int = 0
    last_question: str = ""

    answers_by_step: Dict[str, StepAnswer] = Field(default_factory=dict)
    answers: Dict[str, Any] = Field(default_factory=dict)
    coaching_memory: CoachingMemory = Field(default_factory=CoachingMemory)
    history: List[Dict[str, Any]] = Field(default_factory=list)

    is_completed: bool = False


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