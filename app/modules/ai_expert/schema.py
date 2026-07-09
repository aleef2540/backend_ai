from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any


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

    # GROW Coaching State
    # OPENING -> GOAL -> REALITY -> OPTIONS -> WILL -> SUMMARY -> COMPLETED
    current_coaching_state: str = "OPENING"
    coach_move: Optional[str] = None
    turn_count: int = 0

    # Clarity scoring ใช้กัน AI ถามวน และช่วยให้ข้าม state ได้เมื่อข้อมูลชัดแล้ว
    goal_clarity: int = 0
    reality_clarity: int = 0
    options_count: int = 0
    commitment_clarity: int = 0

    # ใช้ตรวจจับการถามซ้ำ
    last_question_type: Optional[str] = None
    repeated_question_count: int = 0

    # Context Pool สำหรับเก็บข้อมูลที่ Coach สกัดได้แบบละเอียดขึ้น
    extracted_context: Dict[str, Any] = Field(
        default_factory=lambda: {
            "topic": None,
            "goal": {
                "raw": None,
                "desired_outcome": None,
                "success_indicator": None,
            },
            "reality": {
                "current_situation": None,
                "root_causes": [],
                "people_involved": [],
                "constraints": [],
                "user_assumptions": [],
                "impact": None,
            },
            "options": {
                "ideas": [],
                "selected_option": None,
            },
            "will": {
                "action": None,
                "deadline": None,
                "first_step": None,
                "success_measure": None,
                "risk": None,
            },
            "summary": None,
        }
    )


class ChatRequest_aiexpert(BaseModel):
    room_id: Optional[int] = None
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    user_message: str
    course_use: List[Union[int, str]] = Field(default_factory=list)
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
